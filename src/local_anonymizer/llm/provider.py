"""LlmProvider abstraction, LocalApiProvider implementation, and Ollama/Generic discovery/preloading (Phase 6A.1)."""

from __future__ import annotations

import abc
import asyncio
import ipaddress
import json
import logging
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

from local_anonymizer.llm.schema import (
    DiscoveryResult,
    PsModelInfo,
    validate_model_name,
)

logger = logging.getLogger(__name__)

# Named Constants for Local LLM Lifecycle and Security
DEFAULT_OLLAMA_KEEP_ALIVE: str = "5m"
DEFAULT_CONNECT_TIMEOUT: float = 5.0
DEFAULT_READ_TIMEOUT: float = 30.0
DEFAULT_TOTAL_TIMEOUT: float = 60.0
DEFAULT_PRELOAD_TIMEOUT: float = 120.0
MAX_RESPONSE_BYTES: int = 2 * 1024 * 1024  # 2 MB max
PROVIDER_TYPE_OLLAMA: str = "ollama"
PROVIDER_TYPE_GENERIC: str = "generic"


class LlmProvider(abc.ABC):
    """Abstract Base Class for local LLM inference providers."""

    @abc.abstractmethod
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response text from LLM given prompt and optional system prompt."""
        raise NotImplementedError

    @abc.abstractmethod
    async def generate_postcheck(self, prompt: str, system_prompt: str = "", max_tokens: int = 4096) -> str:
        """Generate postcheck response with explicit token reserve and finish_reason validation."""
        raise NotImplementedError("Dieser Provider unterstützt die Phase-6B-Ausgangskontrolle nicht.")

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        raise NotImplementedError


def is_valid_json_mime(content_type: str) -> bool:
    """Check if the provided Content-Type header represents valid JSON."""
    mime_type = content_type.split(";")[0].strip().lower()
    return (
        mime_type in ("application/json", "application/problem+json")
        or bool(re.fullmatch(r"application/[a-z0-9_.\-]+(?:\+[a-z0-9_.\-]+)*\+json", mime_type))
    )


def validate_loopback_url(url_str: str) -> str:
    """
    Validate that the provided base URL is strictly a loopback address without credentials,
    queries, or fragments. Returns normalized base URL or raises ValueError.
    """
    if not url_str or not isinstance(url_str, str):
        raise ValueError("Basis-URL darf nicht leer sein.")

    parsed = urllib.parse.urlsplit(url_str.strip())

    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(f"Ungültiges URL-Schema '{parsed.scheme}': Nur 'http' oder 'https' erlaubt.")

    if parsed.username or parsed.password:
        raise ValueError("URL darf keine Benutzerdaten oder Anmeldeinformationen (Userinfo) enthalten.")

    if parsed.query:
        raise ValueError("URL darf keine Abfrageparameter ('?') enthalten.")

    if parsed.fragment:
        raise ValueError("URL darf keine Fragmente ('#') enthalten.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Ungültiger Hostname in Basis-URL.")

    # Check loopback host
    hostname_clean = hostname.strip().lower()
    if hostname_clean == "localhost":
        is_loopback = True
    else:
        try:
            ip = ipaddress.ip_address(hostname_clean)
            is_loopback = ip.is_loopback
        except ValueError:
            is_loopback = False

    if not is_loopback:
        raise ValueError(
            f"Sicherheitsblockade: Host '{hostname_clean}' ist keine lokale Loopback-Adresse (127.0.0.1 / localhost)."
        )

    # Return normalized clean URL without trailing slash
    clean_path = parsed.path.rstrip("/")
    port_str = f":{parsed.port}" if parsed.port else ""
    host_formatted = f"[{hostname_clean}]" if ":" in hostname_clean and not hostname_clean.startswith("[") else hostname_clean
    return f"{parsed.scheme.lower()}://{host_formatted}{port_str}{clean_path}"


def derive_ollama_base_url(base_url: str) -> str:
    """
    Derive clean Ollama root URL (scheme://host:port) from a validated loopback URL.
    Safely discards path segments like /v1, /chat/completions, or /api.
    """
    clean_url = validate_loopback_url(base_url)
    parsed = urllib.parse.urlsplit(clean_url)
    port_str = f":{parsed.port}" if parsed.port else ""
    hostname = parsed.hostname or "127.0.0.1"
    host_formatted = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    return f"{parsed.scheme.lower()}://{host_formatted}{port_str}"


async def _read_limited_body(
    resp_obj: Any,
    max_bytes: int = MAX_RESPONSE_BYTES,
    chunk_size: int = 8192,
) -> bytes:
    """
    Read response body in chunks up to max_bytes.
    Aborts immediately and raises ValueError if total response size exceeds max_bytes.
    """
    chunks: List[bytes] = []
    total_bytes = 0

    if hasattr(resp_obj, "content") and hasattr(resp_obj.content, "iter_chunked"):
        async for chunk in resp_obj.content.iter_chunked(chunk_size):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError(f"Antwort überschreitet das Limit / Größenlimit von {max_bytes} Bytes.")
            chunks.append(chunk)
    elif hasattr(resp_obj, "content") and hasattr(resp_obj.content, "read"):
        while True:
            chunk = await resp_obj.content.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError(f"Antwort überschreitet das Limit / Größenlimit von {max_bytes} Bytes.")
            chunks.append(chunk)
    else:
        # Fallback for simple mock response objects providing read()
        raw = await resp_obj.read()
        if len(raw) > max_bytes:
            raise ValueError(f"Antwort überschreitet das Limit / Größenlimit von {max_bytes} Bytes.")
        chunks.append(raw)

    return b"".join(chunks)


def _parse_strict_json(body_bytes: bytes) -> Any:
    """Strictly decode UTF-8 without character replacement and parse JSON."""
    try:
        text = body_bytes.decode("utf-8")
    except UnicodeDecodeError as ude:
        raise ValueError("Antwort ist kein gültiges UTF-8.") from ude
    try:
        return json.loads(text)
    except json.JSONDecodeError as jde:
        raise ValueError("Antwort ist kein gültiges JSON.") from jde


def parse_iso_expiry(expires_str: Optional[str]) -> float:
    """
    Parse ISO expiry timestamp with explicit timezone to UNIX epoch float.
    Returns 0.0 if expires_str is missing, unparseable, or naive (missing timezone),
    preventing unverified or infinite readiness.
    """
    if not expires_str or not isinstance(expires_str, str):
        return 0.0
    try:
        from datetime import datetime
        cleaned = expires_str.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            # Naive timestamp without explicit timezone is rejected
            return 0.0
        return dt.timestamp()
    except Exception:
        return 0.0


async def fetch_ollama_models(
    base_url: str,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
) -> DiscoveryResult:
    """
    Query local Ollama server for installed models using native GET /api/tags.
    Enforces loopback, no redirects, proxy bypass, size limit, and cloud tag filtering.
    """
    if aiohttp is None:
        return DiscoveryResult(status="unsupported", message="aiohttp ist nicht installiert.")

    try:
        root_url = derive_ollama_base_url(base_url)
        endpoint = f"{root_url}/api/tags"
    except Exception as e:
        return DiscoveryResult(status="unreachable", message=str(e))

    timeout = aiohttp.ClientTimeout(
        total=connect_timeout + read_timeout,
        connect=connect_timeout,
        sock_read=read_timeout,
    )

    try:
        async with aiohttp.ClientSession(trust_env=False, timeout=timeout) as session:
            async with session.get(endpoint, allow_redirects=False) as resp:
                if resp.status != 200:
                    return DiscoveryResult(
                        status="invalid_response",
                        message=f"Ollama meldete HTTP Status {resp.status}.",
                    )
                raw_ct = resp.headers.get("Content-Type", "")
                if not is_valid_json_mime(raw_ct):
                    return DiscoveryResult(
                        status="invalid_response",
                        message="Ungültiger Content-Type von Ollama empfangen.",
                    )

                try:
                    body_bytes = await _read_limited_body(resp, MAX_RESPONSE_BYTES)
                    data = _parse_strict_json(body_bytes)
                except ValueError as ve:
                    return DiscoveryResult(
                        status="invalid_response",
                        message=f"Ungültige Antwort von Ollama: {ve}",
                    )

                if not isinstance(data, dict) or "models" not in data or not isinstance(data["models"], list):
                    return DiscoveryResult(
                        status="invalid_response",
                        message="Antwortstruktur von Ollama ist ungültig (kein 'models'-Array).",
                    )

                raw_models = data["models"]
                clean_models: List[str] = []
                for m in raw_models:
                    if isinstance(m, dict) and "name" in m and isinstance(m["name"], str):
                        name_val = m["name"].strip()
                        try:
                            valid_name = validate_model_name(name_val)
                            clean_models.append(valid_name)
                        except ValueError:
                            # Filter out invalid names and forbidden cloud tags
                            continue

                if not clean_models:
                    return DiscoveryResult(status="empty", message="Keine lokalen Modelle in Ollama gefunden.")

                return DiscoveryResult(status="success", models=clean_models)
    except asyncio.TimeoutError:
        return DiscoveryResult(status="timeout", message="Zeitüberschreitung bei der Kommunikation mit Ollama.")
    except Exception as e:
        logger.warning(f"Ollama discovery failed: {type(e).__name__}")
        return DiscoveryResult(status="unreachable", message="Ollama-Dienst nicht erreichbar.")


async def fetch_generic_models(
    base_url: str,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
) -> DiscoveryResult:
    """
    Query generic local OpenAI-compatible server using GET /models (or /v1/models).
    """
    if aiohttp is None:
        return DiscoveryResult(status="unsupported", message="aiohttp ist nicht installiert.")

    try:
        clean_url = validate_loopback_url(base_url)
        endpoint = f"{clean_url}/models" if not clean_url.endswith("/models") else clean_url
    except Exception as e:
        return DiscoveryResult(status="unreachable", message=str(e))

    timeout = aiohttp.ClientTimeout(
        total=connect_timeout + read_timeout,
        connect=connect_timeout,
        sock_read=read_timeout,
    )

    try:
        async with aiohttp.ClientSession(trust_env=False, timeout=timeout) as session:
            async with session.get(endpoint, allow_redirects=False) as resp:
                if resp.status != 200:
                    return DiscoveryResult(
                        status="invalid_response",
                        message=f"Server meldete HTTP Status {resp.status}.",
                    )
                raw_ct = resp.headers.get("Content-Type", "")
                if not is_valid_json_mime(raw_ct):
                    return DiscoveryResult(
                        status="invalid_response",
                        message="Ungültiger Content-Type empfangen.",
                    )

                try:
                    body_bytes = await _read_limited_body(resp, MAX_RESPONSE_BYTES)
                    data = _parse_strict_json(body_bytes)
                except ValueError as ve:
                    return DiscoveryResult(
                        status="invalid_response",
                        message=f"Ungültige Antwort vom Server: {ve}",
                    )

                if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], list):
                    return DiscoveryResult(
                        status="invalid_response",
                        message="Antwortstruktur ist ungültig (kein 'data'-Array).",
                    )

                raw_models = data["data"]
                clean_models: List[str] = []
                for m in raw_models:
                    if isinstance(m, dict) and "id" in m and isinstance(m["id"], str):
                        name_val = m["id"].strip()
                        try:
                            valid_name = validate_model_name(name_val)
                            clean_models.append(valid_name)
                        except ValueError:
                            continue

                if not clean_models:
                    return DiscoveryResult(status="empty", message="Keine Modelle auf dem Server gefunden.")

                return DiscoveryResult(status="success", models=clean_models)
    except asyncio.TimeoutError:
        return DiscoveryResult(status="timeout", message="Zeitüberschreitung bei der Kommunikation mit dem Server.")
    except Exception as e:
        logger.warning(f"Generic model fetch failed: {type(e).__name__}")
        return DiscoveryResult(status="unreachable", message="Lokaler Server nicht erreichbar.")


async def preload_ollama_model(
    base_url: str,
    model_name: str,
    keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    preload_timeout: float = DEFAULT_PRELOAD_TIMEOUT,
) -> PsModelInfo:
    """
    Trigger model preload via native POST /api/generate with keep_alive and verify via GET /api/ps.
    Does NOT send any document text or prompt.
    Returns PsModelInfo on exact verified match in /api/ps or raises RuntimeError.
    """
    if aiohttp is None:
        raise ImportError("aiohttp ist nicht installiert.")

    clean_name = validate_model_name(model_name)
    from local_anonymizer.llm.catalog import find_catalog_entry, CatalogError
    target_tag = clean_name
    try:
        cat_entry = find_catalog_entry(clean_name)
        if cat_entry is not None:
            target_tag = cat_entry.tested_tag
    except CatalogError:
        pass

    root_url = derive_ollama_base_url(base_url)
    gen_endpoint = f"{root_url}/api/generate"
    ps_endpoint = f"{root_url}/api/ps"

    timeout = aiohttp.ClientTimeout(
        total=preload_timeout,
        connect=connect_timeout,
        sock_read=preload_timeout,
    )

    payload = {
        "model": target_tag,
        "stream": False,
        "keep_alive": keep_alive,
    }

    try:
        async with aiohttp.ClientSession(trust_env=False, timeout=timeout) as session:
            # 1. Trigger generate preload without prompt/document content
            async with session.post(gen_endpoint, json=payload, allow_redirects=False) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama Vorlade-Anfrage meldete HTTP Status {resp.status}.")
                raw_ct = resp.headers.get("Content-Type", "")
                if not is_valid_json_mime(raw_ct):
                    raise RuntimeError(f"Ungültiger Content-Type bei /api/generate: '{raw_ct}'.")
                gen_body_bytes = await _read_limited_body(resp, MAX_RESPONSE_BYTES)
                gen_data = _parse_strict_json(gen_body_bytes)
                if not isinstance(gen_data, dict):
                    raise RuntimeError("Antwort von /api/generate ist kein gültiges JSON-Objekt.")
                if "error" in gen_data and gen_data["error"]:
                    raise RuntimeError(f"Ollama meldete Fehler beim Vorladen: {gen_data['error']}")
                if gen_data.get("done") is not True or not isinstance(gen_data.get("done"), bool):
                    raise RuntimeError("Ollama Vorlade-Anfrage nicht abgeschlossen ('done' ist nicht True).")
                resp_model = gen_data.get("model")
                if not isinstance(resp_model, str) or not resp_model.strip():
                    raise RuntimeError("Antwort von /api/generate enthält keinen gültigen Modellnamen.")
                resp_model_str = resp_model.strip().lower()
                if resp_model_str not in (clean_name.lower(), target_tag.lower()):
                    raise RuntimeError(
                        f"Modell-Identitätskonflikt bei Vorladung: Angefordert '{target_tag}', aber Ollama antwortete mit '{resp_model}'."
                    )

            # 2. Check /api/ps for active running model
            async with session.get(ps_endpoint, allow_redirects=False) as ps_resp:
                if ps_resp.status != 200:
                    raise RuntimeError(f"Ollama Status-Abfrage (/api/ps) meldete HTTP Status {ps_resp.status}.")

                raw_ct = ps_resp.headers.get("Content-Type", "")
                if not is_valid_json_mime(raw_ct):
                    raise RuntimeError("Ungültiger Content-Type bei /api/ps.")

                body_bytes = await _read_limited_body(ps_resp, MAX_RESPONSE_BYTES)
                ps_data = _parse_strict_json(body_bytes)

                if not isinstance(ps_data, dict) or "models" not in ps_data or not isinstance(ps_data["models"], list):
                    raise RuntimeError("Antwortstruktur von /api/ps ist ungültig.")

                running_models = ps_data["models"]
                clean_lower = clean_name.lower()
                target_lower = target_tag.lower()

                for rm in running_models:
                    if isinstance(rm, dict):
                        rm_name = rm.get("name")
                        rm_model = rm.get("model")
                        rm_name_str = rm_name.strip().lower() if isinstance(rm_name, str) else ""
                        rm_model_str = rm_model.strip().lower() if isinstance(rm_model, str) else ""

                        # EXACT match only: do NOT allow startswith or partial prefixes!
                        if (
                            rm_name_str in (clean_lower, target_lower)
                            or rm_model_str in (clean_lower, target_lower)
                        ):
                            res_name = rm_name.strip() if isinstance(rm_name, str) and rm_name.strip() else clean_name
                            res_model = rm_model.strip() if isinstance(rm_model, str) and rm_model.strip() else clean_name
                            expires_raw = rm.get("expires_at")
                            expires_str = expires_raw.strip() if isinstance(expires_raw, str) and expires_raw.strip() else None
                            return PsModelInfo(
                                name=res_name,
                                model=res_model,
                                size=rm.get("size") if isinstance(rm.get("size"), int) else None,
                                size_vram=rm.get("size_vram") if isinstance(rm.get("size_vram"), int) else None,
                                expires_at=expires_str,
                            )

                raise RuntimeError(f"Modell '{clean_name}' wurde nicht als aktiv in Ollama gemeldet.")
    except asyncio.TimeoutError:
        raise TimeoutError("Zeitüberschreitung beim Laden des Modells in den Speicher.")
    except Exception as e:
        if isinstance(e, (RuntimeError, TimeoutError, ValueError)):
            raise
        logger.warning(f"Preload failed: {type(e).__name__}")
        raise RuntimeError("Verbindung zu Ollama beim Vorladen unterbrochen.")


async def verify_ollama_model_running(
    base_url: str,
    model_name: str,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
) -> Optional[PsModelInfo]:
    """
    Query GET /api/ps on local Ollama server to check if model is actively loaded in memory.
    Returns PsModelInfo if active and unexpired, or None if unloaded/unreachable/expired.
    """
    try:
        clean_url = validate_loopback_url(base_url)
        clean_name = validate_model_name(model_name)
    except Exception:
        return None

    target_tag = clean_name
    try:
        from local_anonymizer.llm.catalog import find_catalog_entry, CatalogError
        cat_entry = find_catalog_entry(clean_name)
        if cat_entry is not None:
            target_tag = cat_entry.tested_tag
    except CatalogError:
        pass

    root_url = derive_ollama_base_url(clean_url)
    ps_endpoint = f"{root_url}/api/ps"

    timeout = aiohttp.ClientTimeout(
        total=connect_timeout + read_timeout,
        connect=connect_timeout,
        sock_read=read_timeout,
    )

    try:
        async with aiohttp.ClientSession(trust_env=False, timeout=timeout) as session:
            async with session.get(ps_endpoint, allow_redirects=False) as ps_resp:
                if ps_resp.status != 200:
                    return None
                raw_ct = ps_resp.headers.get("Content-Type", "")
                if not is_valid_json_mime(raw_ct):
                    return None
                body_bytes = await _read_limited_body(ps_resp, MAX_RESPONSE_BYTES)
                ps_data = _parse_strict_json(body_bytes)
                if not isinstance(ps_data, dict) or "models" not in ps_data or not isinstance(ps_data["models"], list):
                    return None

                clean_lower = clean_name.lower()
                target_lower = target_tag.lower()

                for rm in ps_data["models"]:
                    if isinstance(rm, dict):
                        rm_name = rm.get("name")
                        rm_model = rm.get("model")
                        rm_name_str = rm_name.strip().lower() if isinstance(rm_name, str) else ""
                        rm_model_str = rm_model.strip().lower() if isinstance(rm_model, str) else ""
                        if rm_name_str in (clean_lower, target_lower) or rm_model_str in (clean_lower, target_lower):
                            expires_raw = rm.get("expires_at")
                            expires_str = expires_raw.strip() if isinstance(expires_raw, str) and expires_raw.strip() else None
                            exp_ts = parse_iso_expiry(expires_str)
                            if exp_ts <= 0.0 or time.time() >= exp_ts:
                                return None
                            res_name = rm_name.strip() if isinstance(rm_name, str) and rm_name.strip() else clean_name
                            res_model = rm_model.strip() if isinstance(rm_model, str) and rm_model.strip() else clean_name
                            return PsModelInfo(
                                name=res_name,
                                model=res_model,
                                size=rm.get("size") if isinstance(rm.get("size"), int) else None,
                                size_vram=rm.get("size_vram") if isinstance(rm.get("size_vram"), int) else None,
                                expires_at=expires_str,
                            )
                return None
    except Exception as exc:
        logger.debug(f"verify_ollama_model_running exception: {exc}", exc_info=True)
        return None


async def test_generic_connection(
    base_url: str,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
) -> bool:
    """
    Test generic OpenAI-compatible connection using the validated fetch_generic_models discovery.
    Ensures identical streaming limits, MIME validation, and JSON verification.
    """
    res = await fetch_generic_models(base_url, connect_timeout=connect_timeout, read_timeout=read_timeout)
    if res.status in ("success", "empty"):
        return True
    elif res.status == "timeout":
        raise TimeoutError(res.message or "Zeitüberschreitung bei der Kommunikation mit dem Server.")
    else:
        raise RuntimeError(res.message or "Verbindung zum generischen Server fehlgeschlagen.")


class LocalApiProvider(LlmProvider):
    """
    OpenAI-compatible local API provider (e.g., Ollama, LM Studio).
    Enforces loopback-only endpoints, proxy-bypass, timeouts, size-limits, and session-local concurrency locking.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ):
        if aiohttp is None:
            raise ImportError(
                "aiohttp ist nicht installiert. Bitte installieren Sie local-anonymizer[llm]."
            )

        self.base_url: str = validate_loopback_url(base_url)
        self.model_name: str = validate_model_name(model_name)
        self.connect_timeout: float = float(connect_timeout)
        self.read_timeout: float = float(read_timeout)
        self.total_timeout: float = float(total_timeout)
        self.max_response_bytes: int = int(max_response_bytes)

        self._lock: asyncio.Lock = asyncio.Lock()
        self._session: Optional[Any] = None

    def _get_session(self) -> Any:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.total_timeout,
                connect=self.connect_timeout,
                sock_read=self.read_timeout,
            )
            self._session = aiohttp.ClientSession(
                trust_env=False,  # Bypass system HTTP/HTTPS proxies
                timeout=timeout,
            )
        return self._session

    async def _send_completion_request(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> Tuple[str, Optional[str]]:
        """
        Send a non-streaming chat completion request with response_format: json_object.
        Returns (content, finish_reason).
        """
        if not self.model_name:
            raise ValueError("Kein LLM-Modellname konfiguriert.")

        validate_model_name(self.model_name)

        # Ensure only 1 in-flight request per session instance
        async with self._lock:
            session = self._get_session()

            # Ensure endpoint path points to /chat/completions
            endpoint = f"{self.base_url}/chat/completions" if not self.base_url.endswith("/chat/completions") else self.base_url

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "reasoning_effort": "none",
            }

            headers = {"Content-Type": "application/json"}

            async def _read_response_body(resp_obj: Any) -> str:
                raw_content_type = resp_obj.headers.get("Content-Type", "")
                if not is_valid_json_mime(raw_content_type):
                    raise ValueError(
                        f"Unerwarteter Content-Type '{raw_content_type}': Erwartet wurde 'application/json' oder 'application/*+json'."
                    )

                body_bytes = await _read_limited_body(resp_obj, self.max_response_bytes)
                try:
                    return body_bytes.decode("utf-8")
                except UnicodeDecodeError as ude:
                    raise ValueError("Antwort des LLMs ist kein gültiges UTF-8.") from ude

            try:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    allow_redirects=False,
                ) as resp:
                    if resp.status in (400, 422) and "reasoning_effort" in payload:
                        # Fallback: Retry exactly once without reasoning_effort for servers (e.g. LM Studio) rejecting this parameter
                        payload_retry = dict(payload)
                        payload_retry.pop("reasoning_effort", None)
                        async with session.post(
                            endpoint,
                            json=payload_retry,
                            headers=headers,
                            allow_redirects=False,
                        ) as retry_resp:
                            if retry_resp.status != 200:
                                raise ValueError(f"Lokaler LLM-Provider meldete HTTP Status {retry_resp.status}")
                            raw_body = await _read_response_body(retry_resp)
                    elif resp.status != 200:
                        raise ValueError(f"Lokaler LLM-Provider meldete HTTP Status {resp.status}")
                    else:
                        raw_body = await _read_response_body(resp)

            except asyncio.CancelledError:
                logger.info("LLM-Anfrage wurde abgebrochen (Task Cancelled).")
                if self._session is not None and not getattr(self._session, "closed", True):
                    try:
                        await self._session.close()
                    except Exception:
                        pass
                    self._session = None
                raise
            except asyncio.TimeoutError:
                raise TimeoutError("Zeitüberschreitung bei der Kommunikation mit dem lokalen LLM.")
            except Exception as e:
                # Sanitized error message without leaking prompt or document contents
                if isinstance(e, (ValueError, TypeError)):
                    raise
                msg = str(e)
                if "Antwortgröße" in msg or "HTTP Status" in msg or "Schema version" in msg or "Content-Type" in msg:
                    raise
                logger.warning(f"Lokaler LLM-Aufruf fehlgeschlagen: {type(e).__name__}")
                raise RuntimeError("Verbindung zum lokalen LLM-Dienst fehlgeschlagen oder unterbrochen.")

            # Extract message content and finish_reason from standard OpenAI response structure
            try:
                data = json.loads(raw_body)
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError("Ungültige Antwortstruktur: Keine choices im LLM-Response.")
                choice = choices[0]
                content = choice.get("message", {}).get("content", "")
                finish_reason = choice.get("finish_reason")
                if not content:
                    raise ValueError("Leere Antwort vom lokalen LLM erhalten.")
                return content, finish_reason
            except json.JSONDecodeError:
                raise ValueError("Antwort des lokalen LLMs ist kein gültiges JSON.")

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Send a non-streaming chat completion request with response_format: json_object (Phase 6A Triage).
        Preserves existing 6A behavior.
        """
        content, _ = await self._send_completion_request(prompt, system_prompt=system_prompt, max_tokens=4096)
        return content

    async def generate_postcheck(self, prompt: str, system_prompt: str = "", max_tokens: int = 4096) -> str:
        """
        Send a non-streaming chat completion request for postcheck with explicit max_tokens reserve (Phase 6B).
        Strictly enforces finish_reason == 'stop'; rejects truncation (length) and missing/unexpected status.
        """
        content, finish_reason = await self._send_completion_request(prompt, system_prompt=system_prompt, max_tokens=max_tokens)
        if finish_reason == "length":
            raise ValueError(
                "LLM-Antwort wurde wegen Längenbegrenzung (max_tokens) abgeschnitten – Prüfung unvollständig und abgewiesen."
            )
        if not finish_reason or finish_reason != "stop":
            raise ValueError(
                f"Unerwarteter oder fehlender Abschlussstatus vom LLM-Provider: '{finish_reason}'. Prüfung abgewiesen."
            )
        return content

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
