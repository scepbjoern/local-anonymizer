"""LlmProvider abstraction and LocalApiProvider implementation (Phase 6A)."""

from __future__ import annotations

import abc
import asyncio
import ipaddress
import json
import logging
import re
import urllib.parse
from typing import Any, Dict, Optional

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore


logger = logging.getLogger(__name__)


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


class LocalApiProvider(LlmProvider):
    """
    OpenAI-compatible local API provider (e.g., Ollama, LM Studio).
    Enforces loopback-only endpoints, proxy-bypass, timeouts, size-limits, and session-local concurrency locking.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        total_timeout: float = 60.0,
        max_response_bytes: int = 2 * 1024 * 1024,  # 2 MB max
    ):
        if aiohttp is None:
            raise ImportError(
                "aiohttp ist nicht installiert. Bitte installieren Sie local-anonymizer[llm]."
            )

        self.base_url: str = validate_loopback_url(base_url)
        self.model_name: str = model_name.strip()
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
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
            }

            headers = {"Content-Type": "application/json"}

            try:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    allow_redirects=False,
                ) as resp:
                    if resp.status != 200:
                        raise ValueError(f"Lokaler LLM-Provider meldete HTTP Status {resp.status}")

                    raw_content_type = resp.headers.get("Content-Type", "")
                    mime_type = raw_content_type.split(";")[0].strip().lower()
                    is_valid_json_mime = (
                        mime_type == "application/json"
                        or bool(re.fullmatch(r"application/[a-z0-9_.\-]+(?:\+[a-z0-9_.\-]+)*\+json", mime_type))
                    )
                    if not is_valid_json_mime:
                        raise ValueError(
                            f"Unerwarteter Content-Type '{raw_content_type}': Erwartet wurde 'application/json' oder 'application/*+json'."
                        )

                    # Stream response to enforce max response size limit
                    chunks = []
                    total_bytes = 0
                    async for chunk in resp.content.iter_chunked(8192):
                        total_bytes += len(chunk)
                        if total_bytes > self.max_response_bytes:
                            raise ValueError(
                                f"Antwortgröße des LLMs überschreitet das Limit von {self.max_response_bytes} Bytes."
                            )
                        chunks.append(chunk)

                    raw_body = b"".join(chunks).decode("utf-8", errors="replace")

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
