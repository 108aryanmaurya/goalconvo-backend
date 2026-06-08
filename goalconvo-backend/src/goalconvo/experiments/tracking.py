"""
Reproducibility and experiment tracking: run folders, metadata, prompt fingerprints, artifacts.

Typical layout under ``<experiments_dir>/<run_id>/``::

    metadata.json          # run-level reproducibility + counters
    reproducibility.log    # human-readable append-only log
    dialogues/             # full dialogue JSON per dialogue_id
    artifacts/
      planner/             # pipeline_turns + structured planner slices
      reflection/          # utterance_reflection payloads per dialogue
    metrics/               # optional evaluation / batch stats JSON
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import uuid
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _sha256_short(text: str, n: int = 16) -> str:
    h = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return h[:n]


def prompt_fingerprints() -> Dict[str, str]:
    """
    Content hashes of shipped prompt templates (detect drift across runs/commits).

    Keys are stable logical names; values are short sha256 prefixes of template text.
    """
    try:
        from ..prompts.planner_prompts import PLANNER_PROMPTS, STRUCTURED_DIALOGUE_PLANNER_PROMPTS
        from ..prompts.user_prompts import USER_PROMPTS
        from ..prompts.support_prompts import SUPPORT_PROMPTS
        from ..prompts.reflection_prompts import REFLECTION_PROMPTS
    except Exception as e:  # pragma: no cover
        logger.warning("prompt_fingerprints import failed: %s", e)
        return {"error": _sha256_short(str(e), 12)}

    def pack(prefix: str, d: Dict[str, str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for k, v in (d or {}).items():
            key = f"{prefix}:{k}"
            out[key] = _sha256_short(v or "")
        return out

    fp: Dict[str, str] = {}
    fp.update(pack("planner_experience", PLANNER_PROMPTS))
    fp.update(pack("planner_structured", STRUCTURED_DIALOGUE_PLANNER_PROMPTS))
    fp.update(pack("user", USER_PROMPTS))
    fp.update(pack("support", SUPPORT_PROMPTS))
    fp.update(pack("reflection", REFLECTION_PROMPTS))
    return fp


def _safe_config_dict(config: Any) -> Dict[str, Any]:
    """Serialize Config (or similar dataclass) for JSON with string fallback."""
    if config is None:
        return {}
    if is_dataclass(config):
        try:
            return json.loads(json.dumps(asdict(config), default=str))
        except (TypeError, ValueError):
            pass
    if hasattr(config, "__dict__"):
        try:
            return json.loads(json.dumps(vars(config), default=str))
        except (TypeError, ValueError):
            return {k: str(v) for k, v in vars(config).items()}
    out: Dict[str, Any] = {}
    for f in fields(config):  # type: ignore[arg-type]
        try:
            out[f.name] = json.loads(json.dumps(getattr(config, f.name), default=str))
        except (TypeError, ValueError):
            out[f.name] = str(getattr(config, f.name))
    return out


def generation_hyperparameter_snapshot(config: Any, llm_client: Any) -> Dict[str, Any]:
    """Model id, sampling params, token limits, planner/reflection settings."""
    api = getattr(llm_client, "api_config", None) or {}
    snap = {
        "llm_provider": api.get("provider"),
        "llm_model": api.get("model"),
        "temperature": float(getattr(config, "temperature", 0.7)),
        "top_p": float(getattr(config, "top_p", 0.92)),
        "max_tokens_default": int(getattr(config, "max_tokens", 120)),
        "max_tokens_user_turn": int(getattr(config, "max_tokens_user_turn", 60)),
        "max_tokens_supportbot_turn": int(getattr(config, "max_tokens_supportbot_turn", 120)),
        "max_tokens_planning": int(getattr(config, "max_tokens_planning", 180)),
        "planner_temperature": float(getattr(config, "planner_temperature", 0.1)),
        "planner_top_p": float(getattr(config, "planner_top_p", 0.85)),
        "max_tokens_structured_planner": int(getattr(config, "max_tokens_structured_planner", 320)),
        "reflection_temperature": float(getattr(config, "reflection_temperature", 0.15)),
        "max_tokens_reflection": int(getattr(config, "max_tokens_reflection", 320)),
        "max_turns": int(getattr(config, "max_turns", 15)),
        "min_turns": int(getattr(config, "min_turns", 6)),
        "structured_planner_enabled": bool(getattr(config, "structured_planner_enabled", True)),
        "reflection_on_utterances_enabled": bool(getattr(config, "reflection_on_utterances_enabled", True)),
        "experiment_seed": getattr(config, "experiment_seed", None),
    }
    return snap


def _slug(s: str, max_len: int = 48) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return (s[:max_len] or "run").rstrip("-")


class ExperimentRun:
    """
    One experiment directory: writes metadata, logs, dialogues, planner/reflection slices, metrics.
    """

    def __init__(
        self,
        *,
        run_root: Path,
        run_id: str,
        experiment_name: str,
        config_snapshot: Dict[str, Any],
        gen_snapshot: Dict[str, Any],
        prompt_versions: Dict[str, str],
        started_at: str,
    ):
        self.run_root = run_root
        self.run_id = run_id
        self.experiment_name = experiment_name
        self._config_snapshot = config_snapshot
        self._gen_snapshot = gen_snapshot
        self._prompt_versions = prompt_versions
        self.started_at = started_at
        self._dialogue_count = 0
        self._log_path = run_root / "reproducibility.log"
        self._meta_path = run_root / "metadata.json"
        self.dialogues_dir = run_root / "dialogues"
        self.artifacts_dir = run_root / "artifacts"
        self.planner_dir = self.artifacts_dir / "planner"
        self.reflection_dir = self.artifacts_dir / "reflection"
        self.metrics_dir = run_root / "metrics"
        for p in (
            self.dialogues_dir,
            self.planner_dir,
            self.reflection_dir,
            self.metrics_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    @classmethod
    def start(
        cls,
        config: Any,
        llm_client: Any,
        *,
        experiment_name: str = "",
        run_id: Optional[str] = None,
    ) -> "ExperimentRun":
        base = Path(getattr(config, "experiments_dir", "") or (Path(config.data_dir) / "experiments"))
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short = uuid.uuid4().hex[:8]
        name_part = _slug(experiment_name) if experiment_name else "exp"
        run_id = run_id or f"{stamp}_{name_part}_{short}"
        run_root = base / run_id
        run_root.mkdir(parents=True, exist_ok=False)

        seed = getattr(config, "experiment_seed", None)
        if isinstance(seed, int):
            random.seed(seed)
            try:
                import numpy as np

                np.random.seed(seed)
            except Exception:
                pass

        gen_snap = generation_hyperparameter_snapshot(config, llm_client)
        pfp = prompt_fingerprints()
        started = datetime.now(timezone.utc).isoformat()
        cfg_snap = _safe_config_dict(config)

        inst = cls(
            run_root=run_root,
            run_id=run_id,
            experiment_name=experiment_name or name_part,
            config_snapshot=cfg_snap,
            gen_snapshot=gen_snap,
            prompt_versions=pfp,
            started_at=started,
        )
        inst._write_initial_metadata()
        inst.log_line(
            f"RUN_START run_id={run_id} experiment_name={inst.experiment_name} "
            f"model={gen_snap.get('llm_model')} provider={gen_snap.get('llm_provider')} "
            f"seed={seed!r} temperature={gen_snap.get('temperature')} top_p={gen_snap.get('top_p')}"
        )
        return inst

    def _write_initial_metadata(self) -> None:
        payload: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "started_at": self.started_at,
            "completed_at": None,
            "dialogue_count": 0,
            "generation_hyperparameters": self._gen_snapshot,
            "prompt_versions": self._prompt_versions,
            "config_snapshot": self._config_snapshot,
        }
        with self._meta_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False, default=str)

    def _update_metadata_counters(self) -> None:
        if not self._meta_path.exists():
            return
        try:
            with self._meta_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, OSError):
            data = {}
        data["dialogue_count"] = self._dialogue_count
        data["last_dialogue_at"] = datetime.now(timezone.utc).isoformat()
        with self._meta_path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False, default=str)

    def _write_final_metadata(self) -> None:
        payload: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "started_at": self.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "dialogue_count": self._dialogue_count,
            "generation_hyperparameters": self._gen_snapshot,
            "prompt_versions": self._prompt_versions,
            "config_snapshot": self._config_snapshot,
        }
        with self._meta_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False, default=str)

    def log_line(self, message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}\n"
        with self._log_path.open("a", encoding="utf-8") as fp:
            fp.write(line)

    @staticmethod
    def _reflection_slice_from_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, t in enumerate(turns or []):
            meta = t.get("metadata") if isinstance(t.get("metadata"), dict) else {}
            ur = meta.get("utterance_reflection")
            if ur:
                out.append({"turn_index": i, "role": t.get("role"), "utterance_reflection": ur})
        return out

    def record_dialogue(self, dialogue: Dict[str, Any]) -> None:
        """Persist full dialogue plus planner/reflection artifact files."""
        did = str(dialogue.get("dialogue_id") or "unknown")
        safe = re.sub(r"[^\w\-]+", "_", did)[:120]

        path = self.dialogues_dir / f"{safe}.json"
        with path.open("w", encoding="utf-8") as fp:
            json.dump(dialogue, fp, indent=2, ensure_ascii=False, default=str)

        planner_payload = {
            "dialogue_id": did,
            "pipeline_turns": dialogue.get("pipeline_turns") or [],
        }
        with (self.planner_dir / f"{safe}.json").open("w", encoding="utf-8") as fp:
            json.dump(planner_payload, fp, indent=2, ensure_ascii=False, default=str)

        refl = {
            "dialogue_id": did,
            "reflection_records": self._reflection_slice_from_turns(dialogue.get("turns") or []),
        }
        with (self.reflection_dir / f"{safe}.json").open("w", encoding="utf-8") as fp:
            json.dump(refl, fp, indent=2, ensure_ascii=False, default=str)

        self._dialogue_count += 1
        self._update_metadata_counters()
        self.log_line(f"DIALOGUE_SAVED id={did} path={path.name}")

    def record_metrics(self, name: str, data: Dict[str, Any]) -> None:
        safe = re.sub(r"[^\w\-]+", "_", name)[:80]
        out = self.metrics_dir / f"{safe}.json"
        payload = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "name": name,
            "data": data,
        }
        with out.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False, default=str)
        self.log_line(f"METRICS_SAVED name={name} path={out.name}")

    def finalize(self, summary: Optional[Dict[str, Any]] = None) -> Path:
        """Write final metadata and completion log."""
        if summary:
            with (self.metrics_dir / "run_summary.json").open("w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "run_id": self.run_id,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "summary": summary,
                    },
                    fp,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
        self._write_final_metadata()
        self.log_line(
            f"RUN_END run_id={self.run_id} dialogues_recorded={self._dialogue_count} "
            f"root={self.run_root}"
        )
        return self.run_root


def create_experiment_run(
    config: Any,
    llm_client: Any,
    *,
    experiment_name: str = "",
) -> ExperimentRun:
    """Convenience wrapper for :meth:`ExperimentRun.start`."""
    return ExperimentRun.start(
        config,
        llm_client,
        experiment_name=experiment_name or getattr(config, "experiment_name", "") or "",
    )


def merge_repro_metadata(
    dialogue: Dict[str, Any],
    config: Any,
    llm_client: Any,
    *,
    prompt_versions: Optional[Dict[str, str]] = None,
    experiment_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Attach reproducibility fields under ``metadata["reproducibility"]`` (mutates dialogue in place).
    """
    meta = dialogue.setdefault("metadata", {})
    api = getattr(llm_client, "api_config", None) or {}
    gen = generation_hyperparameter_snapshot(config, llm_client)

    block: Dict[str, Any] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "llm_provider": api.get("provider") or gen.get("llm_provider"),
        "llm_model": api.get("model") or gen.get("llm_model"),
        "temperature": gen.get("temperature"),
        "top_p": gen.get("top_p"),
        "max_tokens_user_turn": gen.get("max_tokens_user_turn"),
        "max_tokens_supportbot_turn": gen.get("max_tokens_supportbot_turn"),
        "planner_temperature": gen.get("planner_temperature"),
        "planner_top_p": gen.get("planner_top_p"),
        "max_tokens_structured_planner": gen.get("max_tokens_structured_planner"),
        "reflection_temperature": gen.get("reflection_temperature"),
        "max_tokens_reflection": gen.get("max_tokens_reflection"),
        "experiment_seed": gen.get("experiment_seed"),
        "prompt_versions": prompt_versions or prompt_fingerprints(),
    }
    if experiment_run_id:
        block["experiment_run_id"] = experiment_run_id
    meta["reproducibility"] = block
    return dialogue
