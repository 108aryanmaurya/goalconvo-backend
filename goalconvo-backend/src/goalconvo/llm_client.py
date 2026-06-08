"""
LLM Client for interfacing with Mistral-7B and other language models via API.

Supports multiple providers with retry logic, rate limiting, and error handling.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Config

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interfacing with language model APIs."""
    
    def __init__(self, config: Config, api_config: Optional[Dict[str, Any]] = None):
        """Initialize the LLM client with configuration."""
        self.config = config
        self.api_config = api_config if api_config is not None else config.get_api_config()
        self.session = self._create_session()
        # Simple in-memory cache for prompt responses (per-process, best-effort)
        # Key: (provider, model, prompt, temperature, top_p, max_tokens, seed_or_-1)
        # Value: generated text
        self._cache: Dict[Tuple[str, str, str, float, float, int, int], str] = {}
        self._cache_max_size: int = 256
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def _maybe_add_seed(self, data: Dict[str, Any]) -> None:
        """Attach OpenAI-style ``seed`` when :attr:`Config.experiment_seed` is set (provider may ignore)."""
        s = getattr(self.config, "experiment_seed", None)
        if isinstance(s, int):
            data["seed"] = s

    def generate_completion(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text completion using the configured LLM.
        
        Args:
            prompt: Input prompt for generation
            temperature: Sampling temperature (overrides config)
            top_p: Top-p sampling parameter (overrides config)
            max_tokens: Maximum tokens to generate (overrides config)
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated text completion
            
        Raises:
            Exception: If API call fails after retries
        """
        # Use config defaults if not provided
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        max_tokens = max_tokens or self.config.max_tokens

        provider = self.api_config["provider"]
        model = self.api_config.get("model", "")
        seed_key = (
            int(self.config.experiment_seed)
            if isinstance(getattr(self.config, "experiment_seed", None), int)
            else -1
        )

        # Build a simple cache key (ignore kwargs for now to keep it small)
        cache_key = (provider, model, prompt, float(temperature), float(top_p), int(max_tokens), seed_key)

        # Return cached response when available
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("LLMClient cache hit for provider=%s model=%s", provider, model)
            return cached

        if provider == "openrouter":
            result = self._call_openrouter_api(prompt, temperature, top_p, max_tokens, **kwargs)
        elif provider == "groq":
            result = self._call_groq_api(prompt, temperature, top_p, max_tokens, **kwargs)
        elif provider == "deepseek":
            result = self._call_deepseek_api(prompt, temperature, top_p, max_tokens, **kwargs)
        elif provider == "gemini":
            result = self._call_gemini_api(prompt, temperature, top_p, max_tokens, **kwargs)
        elif provider == "ollama":
            result = self._call_ollama_api(prompt, temperature, top_p, max_tokens, **kwargs)
        elif provider == "mistral":
            result = self._call_mistral_api(prompt, temperature, top_p, max_tokens, **kwargs)
        elif provider == "openai":
            result = self._call_openai_api(prompt, temperature, top_p, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Update cache with simple eviction when over capacity
        try:
            if len(self._cache) >= self._cache_max_size:
                # Pop an arbitrary item (insertion order not guaranteed before 3.7,
                # but for best-effort caching this is acceptable)
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result
        except Exception as cache_error:
            logger.debug("LLMClient cache update failed: %s", cache_error)

        return result
    
    def _call_mistral_api(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Call Mistral API (via Together AI or similar)."""
        url = f"{self.api_config['api_base']}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.api_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs
        }
        self._maybe_add_seed(data)

        try:
            response = self.session.post(
                url,
                headers=headers,
                json=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Mistral API call failed: {e}")
            raise Exception(f"API call failed: {e}")
    
    def _call_ollama_api(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Call Ollama API (local) using native API format."""
        api_base = self.api_config['api_base'].rstrip('/')
        
        # Use native Ollama API format (faster and more reliable)
        url = f"{api_base}/api/generate"
        headers = {"Content-Type": "application/json"}
        
        # Optimize: Use streaming for faster responses (get first tokens immediately)
        # But for simplicity, we'll use non-streaming with optimized settings
        data = {
            "model": self.api_config["model"],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                # Optimize: Add num_ctx to limit context window for faster processing
                "num_ctx": min(512, len(prompt.split()) * 2),  # Limit context window
            }
        }
        
        try:
            # Use longer timeout for local Ollama models (phi2:mini can be slow)
            timeout = getattr(self.config, 'ollama_timeout', self.config.timeout)
            logger.debug(f"Calling Ollama API with timeout={timeout}s, max_tokens={max_tokens}")
            
            response = self.session.post(
                url,
                headers=headers,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if "response" in result:
                return result["response"].strip()
            else:
                raise Exception(f"Unexpected response format: {result}")
            
        except requests.exceptions.Timeout as e:
            logger.warning(f"Ollama API call timed out after {timeout}s. Model may be slow. Retrying with reduced max_tokens...")
            # Retry with even fewer tokens if timeout occurs
            if max_tokens > 20:
                try:
                    data["options"]["num_predict"] = 20
                    logger.debug(f"Retrying with max_tokens=20")
                    response = self.session.post(url, headers=headers, json=data, timeout=timeout)
                    response.raise_for_status()
                    result = response.json()
                    if "response" in result:
                        return result["response"].strip()
                except Exception as retry_e:
                    logger.error(f"Retry also failed: {retry_e}")
            logger.error(f"Ollama API call timed out after {timeout}s. Model may be too slow or stuck.")
            logger.error(f"Consider: 1) Reducing max_tokens (current: {max_tokens}), 2) Using a faster model, 3) Checking Ollama server status")
            raise Exception(f"API call timed out after {timeout}s: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API call failed: {e}")
            logger.error(f"Check if Ollama is running: curl {api_base}/api/tags")
            raise Exception(f"API call failed: {e}")
    
    def _call_gemini_api(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Call Google Gemini API."""
        api_base = self.api_config['api_base'].rstrip('/')
        model = self.api_config['model']
        api_key = self.api_config['api_key']
        
        # Gemini API endpoint format: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}
        url = f"{api_base}/models/{model}:generateContent"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Gemini API request format
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_tokens,
            }
        }
        
        try:
            # Add API key as query parameter
            params = {"key": api_key}
            
            logger.debug(f"Calling Gemini API with model={model}, max_tokens={max_tokens}")
            
            response = self.session.post(
                url,
                headers=headers,
                json=data,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response text from Gemini API response format
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"].strip()
            
            raise Exception(f"Unexpected Gemini API response format: {result}")
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Gemini API call timed out after {self.config.timeout}s")
            raise Exception(f"API call timed out: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API call failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    logger.error(f"Response: {e.response.text}")
                except:
                    pass
            raise Exception(f"API call failed: {e}")
    
    def _call_groq_api(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Call Groq API (OpenAI-compatible). Uses max_completion_tokens per Groq docs."""
        base = self.api_config["api_base"].rstrip("/")
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_config['api_key']}",
            "Content-Type": "application/json"
        }
        # Groq prefers max_completion_tokens over deprecated max_tokens; only send supported params
        data = {
            "model": self.api_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_completion_tokens": max_tokens,
        }
        self._maybe_add_seed(data)
        try:
            response = self.session.post(
                url,
                headers=headers,
                json=data,
                timeout=self.config.timeout
            )
            if not response.ok:
                try:
                    err_body = response.json()
                    if response.status_code == 429:
                        logger.warning(
                            "Groq rate limit (429). Use another provider by unsetting GROQ_API_KEY in .env "
                            "(e.g. use DeepSeek or OpenAI), or wait and retry. Error: %s", err_body
                        )
                    else:
                        logger.error("Groq API error response: %s", err_body)
                except Exception:
                    if response.status_code == 429:
                        logger.warning("Groq rate limit (429). Unset GROQ_API_KEY to use another provider.")
                    else:
                        logger.error("Groq API call failed: %s %s", response.status_code, response.text[:500])
                response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            logger.error("Groq API call failed: %s", e)
            raise Exception(f"API call failed: {e}") from e

    def _call_openrouter_api(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Call OpenRouter.ai API (OpenAI-compatible unified gateway)."""
        base = self.api_config["api_base"].rstrip("/")
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_config['api_key']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.api_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs
        }
        self._maybe_add_seed(data)
        try:
            response = self.session.post(
                url,
                headers=headers,
                json=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            logger.error("OpenRouter API call failed: %s", e)
            raise Exception(f"API call failed: {e}") from e

    def _call_deepseek_api(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Call DeepSeek API (OpenAI-compatible)."""
        base = self.api_config["api_base"].rstrip("/")
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_config['api_key']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.api_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs
        }
        self._maybe_add_seed(data)
        try:
            response = self.session.post(
                url,
                headers=headers,
                json=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            logger.error("DeepSeek API call failed: %s", e)
            raise Exception(f"API call failed: {e}") from e

    def _call_openai_api(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Call OpenAI API."""
        url = f"{self.api_config['api_base']}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.api_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs
        }
        self._maybe_add_seed(data)

        try:
            response = self.session.post(
                url,
                headers=headers,
                json=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise Exception(f"API call failed: {e}")
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate completions for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            List of generated text completions
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                result = self.generate_completion(
                    prompt, temperature, top_p, max_tokens, **kwargs
                )
                results.append(result)
                
                # Add small delay between requests to avoid rate limiting
                if i < len(prompts) - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Failed to generate completion for prompt {i}: {e}")
                results.append("")  # Add empty string for failed generations
        
        return results
    
    def test_connection(self) -> bool:
        """
        Test the API connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_prompt = "Hello, this is a test. Please respond with 'Connection successful.'"
            response = self.generate_completion(test_prompt, max_tokens=10)
            return "successful" in response.lower()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
