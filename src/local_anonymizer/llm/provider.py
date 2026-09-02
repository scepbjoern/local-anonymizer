"""LlmProvider abstraction, LocalApiProvider implementation, and Ollama/Generic discovery/preloading (Phase 6A.1)."""

from __future__ import annotations

import abc
import asyncio
import ipaddress
import json
import logging
import re
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

                body_bytes = await resp.read()
                if len(body_bytes) > MAX_RESPONSE_BYTES:
                    return DiscoveryResult(
                        status="invalid_response",
                        message="Antwort von Ollama überschreitet das Größenlimit.",
                    )

                data = json.loads(body_bytes.decode("utf-8", errors="replace"))
                raw_models = data.get("models", [])
                if not isinstance(raw_models, list):
                    return DiscoveryResult(
                        status="invalid_response",
                        message="Antwortstruktur von Ollama ist ungültig.",
                    )

                clean_models: List[str] = []
                for m in raw_models:
                    if isinstance(m, dict) and "name" in m:
                        name_val = str(m["name"]).strip()
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

                body_bytes = await resp.read()
                if len(body_bytes) > MAX_RESPONSE_BYTES:
                    return DiscoveryResult(
                        status="invalid_response",
                        message="Antwort überschreitet das Größenlimit.",
                    )

                data = json.loads(body_bytes.decode("utf-8", errors="replace"))
                raw_models = data.get("data", [])
                if not isinstance(raw_models, list):
                    return DiscoveryResult(
                        status="invalid_response",
                        message="Antwortstruktur ist ungültig (kein 'data'-Array).",
                    )

                clean_models: List[str] = []
                for m in raw_models:
                    if isinstance(m, dict) and "id" in m:
                        name_val = str(m["id"]).strip()
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
    Returns PsModelInfo on verified match in /api/ps or raises RuntimeError.
    """
    if aiohttp is None:
        raise ImportError("aiohttp ist nicht installiert.")

    clean_name = validate_model_name(model_name)
    root_url = derive_ollama_base_url(base_url)
    gen_endpoint = f"{root_url}/api/generate"
    ps_endpoint = f"{root_url}/api/ps"

    timeout = aiohttp.ClientTimeout(
        total=preload_timeout,
        connect=connect_timeout,
        sock_read=preload_timeout,
    )

    payload = {
        "model": clean_name,
        "stream": False,
        "keep_alive": keep_alive,
    }

    try:
        async with aiohttp.ClientSession(trust_env=False, timeout=timeout) as session:
            # 1. Trigger generate preload without prompt/document content
            async with session.post(gen_endpoint, json=payload, allow_redirects=False) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama Vorlade-Anfrage meldete HTTP Status {resp.status}.")

            # 2. Check /api/ps for active running model
            async with session.get(ps_endpoint, allow_redirects=False) as ps_resp:
                if ps_resp.status != 200:
                    raise RuntimeError(f"Ollama Status-Abfrage (/api/ps) meldete HTTP Status {ps_resp.status}.")

                raw_ct = ps_resp.headers.get("Content-Type", "")
                if not is_valid_json_mime(raw_ct):
                    raise RuntimeError("Ungültiger Content-Type bei /api/ps.")

                body_bytes = await ps_resp.read()
                if len(body_bytes) > MAX_RESPONSE_BYTES:
                    raise RuntimeError("Antwort von /api/ps überschreitet das Größenlimit.")

                ps_data = json.loads(body_bytes.decode("utf-8", errors="replace"))
                running_models = ps_data.get("models", [])

                clean_lower = clean_name.lower()
                for rm in running_models:
                    if isinstance(rm, dict):
                        rm_name = str(rm.get("name", "")).strip().lower()
                        rm_model = str(rm.get("model", "")).strip().lower()
                        if (
                            rm_name == clean_lower
                            or rm_model == clean_lower
                            or rm_name.startswith(f"{clean_lower}:")
                            or clean_lower.startswith(f"{rm_name}:")
                        ):
                            return PsModelInfo(
                                name=str(rm.get("name", clean_name)),
                                model=str(rm.get("model", clean_name)),
                                size=rm.get("size"),
                                size_vram=rm.get("size_vram"),
                                expires_at=rm.get("expires_at"),
                            )

                raise RuntimeError(f"Modell '{clean_name}' wurde nicht als aktiv in Ollama gemeldet.")
    except asyncio.TimeoutError:
        raise TimeoutError("Zeitüberschreitung beim Laden des Modells in den Speicher.")
    except Exception as e:
        if isinstance(e, (RuntimeError, TimeoutError, ValueError)):
            raise
        logger.warning(f"Preload failed: {type(e).__name__}")
        raise RuntimeError("Verbindung zu Ollama beim Vorladen unterbrochen.")


async def test_generic_connection(
    base_url: str,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
) -> bool:
    """
    Test generic OpenAI-compatible connection with a lightweight GET /models call.
    Does NOT guarantee loaded model weights.
    """
    if aiohttp is None:
        raise ImportError("aiohttp ist nicht installiert.")

    clean_url = validate_loopback_url(base_url)
    endpoint = f"{clean_url}/models" if not clean_url.endswith("/models") else clean_url

    timeout = aiohttp.ClientTimeout(
        total=connect_timeout + read_timeout,
        connect=connect_timeout,
        sock_read=read_timeout,
    )

    try:
        async with aiohttp.ClientSession(trust_env=False, timeout=timeout) as session:
            async with session.get(endpoint, allow_redirects=False) as resp:
                if resp.status == 200:
                    return True
                raise RuntimeError(f"Server meldete HTTP Status {resp.status}.")
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError("Verbindung zum generischen Server fehlgeschlagen.")


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

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Send a non-streaming chat completion request with response_format: json_object.
        Uses session-local asyncio.Lock to avoid interleaving requests on the same provider instance.
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
                "max_tokens": 4096,
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

                # Stream response to enforce max response size limit
                chunks = []
                total_bytes = 0
                async for chunk in resp_obj.content.iter_chunked(8192):
                    total_bytes += len(chunk)
                    if total_bytes > self.max_response_bytes:
                        raise ValueError(
                            f"Antwortgröße des LLMs überschreitet das Limit von {self.max_response_bytes} Bytes."
                        )
                    chunks.append(chunk)

                return b"".join(chunks).decode("utf-8", errors="replace")

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
                msg = str(e)
                if "Antwortgröße" in msg or "HTTP Status" in msg or "Schema version" in msg or "Content-Type" in msg:
                    raise
                logger.warning(f"Lokaler LLM-Aufruf fehlgeschlagen: {type(e).__name__}")
                raise RuntimeError("Verbindung zum lokalen LLM-Dienst fehlgeschlagen oder unterbrochen.")

            # Extract message content from standard OpenAI response structure
            try:
                data = json.loads(raw_body)
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError("Ungültige Antwortstruktur: Keine choices im LLM-Response.")
                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError("Leere Antwort vom lokalen LLM erhalten.")
                return content
            except json.JSONDecodeError:
                raise ValueError("Antwort des lokalen LLMs ist kein gültiges JSON.")

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
