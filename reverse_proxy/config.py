"""Configuration management for the reverse proxy server."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

CONFIG_ENV_VAR = "REVERSE_PROXY_CONFIG_PATH"
DEFAULT_CONFIG_FILENAME = "config.json"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {field_name}") from exc


def _coerce_string_sequence(value: Any, *, field_name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        trimmed = value.strip()
        return (trimmed,) if trimmed else ()
    if isinstance(value, (list, tuple, set)):
        seen = set()
        result = []
        for item in value:
            if item is None:
                continue
            item_str = str(item).strip()
            if not item_str or item_str in seen:
                continue
            seen.add(item_str)
            result.append(item_str)
        return tuple(result)
    raise ValueError(f"{field_name} must be a sequence of strings")


@dataclass(frozen=True)
class ServerConfig:
    """Server configuration settings."""

    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ServerConfig":
        defaults = cls()
        if not data:
            return defaults
        host = str(data.get("host", defaults.host))
        port = _coerce_int(data.get("port", defaults.port), field_name="server.port")
        reload_flag = _parse_bool(data.get("reload", defaults.reload))
        log_level = str(data.get("log_level", defaults.log_level))
        return cls(host=host, port=port, reload=reload_flag, log_level=log_level)


@dataclass(frozen=True)
class AuthConfig:
    """Authentication configuration settings."""

    miner_hotkey: str
    allowed_delta_ms: int = 8000
    cache_duration: int = 3600
    chain_endpoint: str = "wss://entrypoint-finney.opentensor.ai:443"
    allowed_senders: Tuple[str, ...] = field(default_factory=tuple)
    self_debug_key: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "AuthConfig":
        if not data or not data.get("miner_hotkey"):
            raise ValueError("Configuration missing 'auth.miner_hotkey'")
        miner_hotkey = str(data.get("miner_hotkey", "")).strip()
        if not miner_hotkey:
            raise ValueError("Configuration missing 'auth.miner_hotkey'")
        allowed_delta_ms = _coerce_int(
            data.get("allowed_delta_ms", cls.allowed_delta_ms),
            field_name="auth.allowed_delta_ms",
        )
        cache_duration = _coerce_int(
            data.get("cache_duration", cls.cache_duration),
            field_name="auth.cache_duration",
        )
        chain_endpoint = str(data.get("chain_endpoint", cls.chain_endpoint))
        allowed_senders = _coerce_string_sequence(
            data.get("allowed_senders", ()), field_name="auth.allowed_senders"
        )
        debug_key_raw = data.get("self-debug-key") or data.get("self_debug_key")
        debug_key = str(debug_key_raw).strip() if debug_key_raw else None
        return cls(
            miner_hotkey=miner_hotkey,
            allowed_delta_ms=allowed_delta_ms,
            cache_duration=cache_duration,
            chain_endpoint=chain_endpoint,
            allowed_senders=allowed_senders,
            self_debug_key=debug_key,
        )


@dataclass(frozen=True)
class ServiceConfig:
    """Internal service configuration."""

    inference_server_url: str = "http://localhost:8091"

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ServiceConfig":
        defaults = cls()
        if not data:
            return defaults
        inference_url = str(data.get("inference_server_url", defaults.inference_server_url))
        return cls(inference_server_url=inference_url)


@dataclass(frozen=True)
class Config:
    """Complete application configuration."""

    server: ServerConfig
    auth: AuthConfig
    services: ServiceConfig
    source_path: Path = field(default=DEFAULT_CONFIG_PATH, repr=False)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        source_path: Optional[Path] = None,
    ) -> "Config":
        server_cfg = ServerConfig.from_dict(data.get("server"))
        auth_cfg = AuthConfig.from_dict(data.get("auth"))
        services_cfg = ServiceConfig.from_dict(data.get("services"))
        return cls(
            server=server_cfg,
            auth=auth_cfg,
            services=services_cfg,
            source_path=source_path or DEFAULT_CONFIG_PATH,
        )

    def masked_dict(self) -> Dict[str, Any]:
        """Return configuration dictionary with sensitive fields masked."""
        masked_auth = {
            **self.auth.__dict__,
            "miner_hotkey": _mask_secret(self.auth.miner_hotkey),
            "allowed_senders": list(self.auth.allowed_senders),
        }
        if masked_auth.get("self_debug_key"):
            masked_auth["self_debug_key"] = _mask_secret(self.auth.self_debug_key or "")
        return {
            "server": dict(self.server.__dict__),
            "auth": masked_auth,
            "services": dict(self.services.__dict__),
            "source_path": str(self.source_path),
        }


def _mask_secret(value: str) -> str:
    trimmed = value.strip()
    if len(trimmed) <= 8:
        return "***"
    return f"{trimmed[:4]}...{trimmed[-4:]}"


def resolve_config_path(path: Optional[str] = None) -> Path:
    if path:
        candidate = Path(path).expanduser()
    else:
        env_path = os.getenv(CONFIG_ENV_VAR)
        candidate = Path(env_path).expanduser() if env_path else DEFAULT_CONFIG_PATH
    if candidate.is_dir():
        candidate = candidate / DEFAULT_CONFIG_FILENAME
    return candidate


def load_config(path: Optional[str] = None) -> Config:
    resolved_path = resolve_config_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {resolved_path}")
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse configuration file {resolved_path}: {exc}") from exc
    if not isinstance(raw_data, Mapping):
        raise ValueError("Configuration root must be a JSON object")
    return Config.from_dict(raw_data, source_path=resolved_path)
