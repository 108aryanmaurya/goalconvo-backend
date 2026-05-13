"""
Configuration module for GoalConvo framework.

Contains all hyperparameters and settings from the research paper.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration class containing all hyperparameters and settings."""

    # OpenRouter.ai (unified API, OpenAI-compatible)
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_api_base: str = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")

    # Groq (OpenAI-compatible API)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_api_base: str = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # DeepSeek (OpenAI-compatible API)
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_api_base: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Google Gemini API configuration (disabled by default; requires explicit key)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_api_base: str = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    # API Configuration
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    mistral_api_base: str = os.getenv("MISTRAL_API_BASE", "https://api.together.xyz/v1")
    mistral_model: str = os.getenv("MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
    
    # Alternative API providers
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Ollama (local) configuration
    ollama_enabled: bool = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
    ollama_api_base: str = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral")
    # Keep local models responsive by default; can be increased via env
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "240"))  # 60 seconds for local models (phi2:mini can be slow)
    
    # Generation hyperparameters (from paper; lower temp = more focused, on-goal dialogues)
    temperature: float = float(os.getenv("TEMPERATURE", "0.65"))  # 0.65 for coherent, on-goal turns (was 0.75)
    top_p: float = float(os.getenv("TOP_P", "0.92"))
    # Use smaller defaults for faster responses; can be overridden via env
    max_tokens: int = int(os.getenv("MAX_TOKENS", "120"))
    max_turns: int = int(os.getenv("MAX_TURNS", "15"))  # Maximum turns per dialogue
    min_turns: int = int(os.getenv("MIN_TURNS", "6"))  # Minimum turns per dialogue (enforced)
    # Per-turn length limits for dialogue simulation (shorter = more concise turns)
    max_tokens_user_turn: int = int(os.getenv("MAX_TOKENS_USER_TURN", "60"))
    max_tokens_supportbot_turn: int = int(os.getenv("MAX_TOKENS_SUPPORTBOT_TURN", "120"))  # 120 for fuller confirmations
    
    # Few-shot settings (use 3-5 for better patterns; seed hub has strong examples)
    few_shot_examples: int = int(os.getenv("FEW_SHOT_EXAMPLES", "4"))
    
    # Quality filtering
    quality_threshold: float = float(os.getenv("QUALITY_THRESHOLD", "0.7"))
    discard_rate: float = float(os.getenv("DISCARD_RATE", "0.1"))
    # When True, rejected dialogues get one LLM improvement attempt and re-judged (improves acceptance quality)
    quality_improve_on_fail: bool = os.getenv("QUALITY_IMPROVE_ON_FAIL", "true").lower() in ("true", "1", "yes")
    
    # Generation settings
    max_dialogues: int = int(os.getenv("MAX_DIALOGUES", "20000"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    
    # Data paths - will be set in __post_init__ to use absolute paths
    data_dir: str = field(default="")
    synthetic_dir: str = field(default="")
    multiwoz_dir: str = field(default="")
    few_shot_hub_dir: str = field(default="")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "./logs/goalconvo.log")
    
    # Domains (from MultiWOZ)
    domains: List[str] = field(default_factory=lambda: ["hotel",
    #  "restaurant", "taxi", "train", "attraction"
     ])
    
    # API retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    # Overall API timeout (seconds) for upstream LLM calls
    timeout: int = int(os.getenv("API_TIMEOUT", "60"))
    
    # Prompt truncation (keep full goal + last K turns; avoid dropping mid-dialogue context)
    prompt_max_words: int = int(os.getenv("PROMPT_MAX_WORDS", "1000"))
    prompt_instruction_words: int = int(os.getenv("PROMPT_INSTRUCTION_WORDS", "250"))
    prompt_last_k_turns: int = int(os.getenv("PROMPT_LAST_K_TURNS", "6"))  # last 6 turns = 3 exchanges

    # Multi-agent capabilities: memory, tool stubs, planning, RL-lite telemetry
    agent_memory_enabled: bool = os.getenv("AGENT_MEMORY_ENABLED", "true").lower() in ("true", "1", "yes")
    agent_tools_enabled: bool = os.getenv("AGENT_TOOLS_ENABLED", "true").lower() in ("true", "1", "yes")
    agent_planning_enabled: bool = os.getenv("AGENT_PLANNING_ENABLED", "true").lower() in ("true", "1", "yes")
    rl_lite_enabled: bool = os.getenv("RL_LITE_ENABLED", "true").lower() in ("true", "1", "yes")
    rl_goal_weight: float = float(os.getenv("RL_GOAL_WEIGHT", "0.6"))
    rl_coherence_weight: float = float(os.getenv("RL_COHERENCE_WEIGHT", "0.4"))
    max_tokens_memory_refresh: int = int(os.getenv("MAX_TOKENS_MEMORY_REFRESH", "256"))
    max_tokens_planning: int = int(os.getenv("MAX_TOKENS_PLANNING", "180"))

    # Evaluation settings
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    diversity_metrics: List[str] = field(default_factory=lambda: ["distinct-1", "distinct-2", "self-bleu"])
    
    def __post_init__(self):
        """Validate configuration after initialization and set data paths."""
        # Set data paths relative to goalconvo-2 directory (where config.py is located)
        # Use resolve() to get absolute paths regardless of current working directory
        base_dir = Path(__file__).parent.parent.parent.resolve()  # Go up from src/goalconvo/config.py to goalconvo-2/
        
        if not self.data_dir or self.data_dir == "":
            self.data_dir = os.getenv("DATA_DIR", str(base_dir / "data"))
        if not self.synthetic_dir or self.synthetic_dir == "":
            self.synthetic_dir = os.getenv("SYNTHETIC_DIR", str(base_dir / "data" / "synthetic"))
        if not self.multiwoz_dir or self.multiwoz_dir == "":
            self.multiwoz_dir = os.getenv("MULTIWOZ_DIR", str(base_dir / "data" / "multiwoz"))
        if not self.few_shot_hub_dir or self.few_shot_hub_dir == "":
            self.few_shot_hub_dir = os.getenv("FEW_SHOT_HUB_DIR", str(base_dir / "data" / "few_shot_hub"))
        
        if not self.ollama_enabled and not self.mistral_api_key and not self.openai_api_key and not self.gemini_api_key and not self.deepseek_api_key and not self.groq_api_key and not self.openrouter_api_key:
            raise ValueError("Set at least one: OPENROUTER_API_KEY, GROQ_API_KEY, OLLAMA_ENABLED=true, DEEPSEEK_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY, or OPENAI_API_KEY")
        
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("Top-p must be between 0 and 1")
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration for the selected provider.
        
        Priority order:
        1. Groq - if API key available
        2. DeepSeek - if API key available
        3. OpenRouter.ai - if API key available
        4. Ollama (local) - if enabled
        5. Gemini - if API key available
        6. OpenAI (ChatGPT) - if API key available
        7. Mistral - if API key available
        """
        if self.openrouter_api_key:
            return {
                "api_key": self.openrouter_api_key,
                "api_base": self.openrouter_api_base,
                "model": self.openrouter_model,
                "provider": "openrouter"
            }
        # Priority 1: Groq
        if self.groq_api_key:
            return {
                "api_key": self.groq_api_key,
                "api_base": self.groq_api_base,
                "model": self.groq_model,
                "provider": "groq"
            }
        # Priority 2: DeepSeek
        if self.deepseek_api_key:
            return {
                "api_key": self.deepseek_api_key,
                "api_base": self.deepseek_api_base,
                "model": self.deepseek_model,
                "provider": "deepseek"
            }
        # Priority 3: OpenRouter.ai

        # Priority 4: Ollama (local, if enabled)
        if self.ollama_enabled:
            return {
                "api_key": "",  # Ollama doesn't require API key
                "api_base": self.ollama_api_base,
                "model": self.ollama_model,
                "provider": "ollama"
            }
        # Priority 5: Gemini
        elif self.gemini_api_key:
            return {
                "api_key": self.gemini_api_key,
                "api_base": self.gemini_api_base,
                "model": self.gemini_model,
                "provider": "gemini"
            }
        # Priority 6: OpenAI (ChatGPT)
        elif self.openai_api_key:
            return {
                "api_key": self.openai_api_key,
                "api_base": self.openai_api_base,
                "model": self.openai_model,
                "provider": "openai"
            }
        # Priority 7: Mistral
        elif self.mistral_api_key:
            return {
                "api_key": self.mistral_api_key,
                "api_base": self.mistral_api_base,
                "model": self.mistral_model,
                "provider": "mistral"
            }
        else:
            raise ValueError("No valid API configuration found. Set OPENROUTER_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, OLLAMA_ENABLED=true, MISTRAL_API_KEY, or OPENAI_API_KEY")

    def _resolve_provider_config(self, provider_name: str, model_override: str = "") -> Dict[str, Any]:
        """Resolve API config for a specific provider name."""
        provider = (provider_name or "").strip().lower()
        model_override = (model_override or "").strip()

        if provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when provider is 'gemini'")
            return {
                "api_key": self.gemini_api_key,
                "api_base": self.gemini_api_base,
                "model": model_override or self.gemini_model,
                "provider": "gemini",
            }
        if provider == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY is required when provider is 'openrouter'")
            return {
                "api_key": self.openrouter_api_key,
                "api_base": self.openrouter_api_base,
                "model": model_override or self.openrouter_model,
                "provider": "openrouter",
            }
        if provider == "claude":
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY is required when provider is 'claude' (via OpenRouter)")
            return {
                "api_key": self.openrouter_api_key,
                "api_base": self.openrouter_api_base,
                "model": model_override or os.getenv("EVALUATION_MODEL", "anthropic/claude-3.5-sonnet"),
                "provider": "openrouter",
            }
        if provider == "deepseek":
            if not self.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY is required when provider is 'deepseek'")
            return {
                "api_key": self.deepseek_api_key,
                "api_base": self.deepseek_api_base,
                "model": model_override or self.deepseek_model,
                "provider": "deepseek",
            }
        if provider == "groq":
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY is required when provider is 'groq'")
            return {
                "api_key": self.groq_api_key,
                "api_base": self.groq_api_base,
                "model": model_override or self.groq_model,
                "provider": "groq",
            }
        if provider == "ollama":
            if not self.ollama_enabled:
                raise ValueError("OLLAMA_ENABLED=true is required when provider is 'ollama'")
            return {
                "api_key": "",
                "api_base": self.ollama_api_base,
                "model": model_override or self.ollama_model,
                "provider": "ollama",
            }
        if provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when provider is 'openai'")
            return {
                "api_key": self.openai_api_key,
                "api_base": self.openai_api_base,
                "model": model_override or self.openai_model,
                "provider": "openai",
            }
        if provider == "mistral":
            if not self.mistral_api_key:
                raise ValueError("MISTRAL_API_KEY is required when provider is 'mistral'")
            return {
                "api_key": self.mistral_api_key,
                "api_base": self.mistral_api_base,
                "model": model_override or self.mistral_model,
                "provider": "mistral",
            }
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported: gemini, claude, openrouter, deepseek, groq, ollama, openai, mistral"
        )

    def get_generation_api_config(self) -> Dict[str, Any]:
        """Get API configuration for dialogue generation."""
        provider = os.getenv("GENERATION_PROVIDER", "groq")
        model = os.getenv("GENERATION_MODEL", "")
        return self._resolve_provider_config(provider, model_override=model)

    def get_evaluation_api_config(self) -> Dict[str, Any]:
        """Get API configuration for dialogue evaluation/judging."""
        provider = os.getenv("EVALUATION_PROVIDER", "groq")
        model = os.getenv("EVALUATION_MODEL", "")
        return self._resolve_provider_config(provider, model_override=model)
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Get parameters for text generation."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "max_turns": self.max_turns,
            "min_turns": self.min_turns
        }
    
    def get_quality_params(self) -> Dict[str, Any]:
        """Get parameters for quality filtering."""
        return {
            "threshold": self.quality_threshold,
            "discard_rate": self.discard_rate,
            "min_turns": self.min_turns,
            "max_turns": self.max_turns
        }
