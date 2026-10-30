import asyncio
import json
import logging
import re
import httpx
from typing import Optional, Dict, Any
from app import config

logger = logging.getLogger("safefix.models")

class LLMClient:
    """Unified LLM client supporting both Groq (dev) and NVIDIA Token Factory / Nemotron (hackathon).
    
    Guarantees:
    - Never silently substitutes simulated responses for real runs; simulation is restricted strictly to explicitly labeled demo mode.
    - Handles provider-specific rate limits, context clipping, and backoff retries.
    """

    def __init__(
        self,
        provider: str = config.DEFAULT_PROVIDER,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        ultra_model: Optional[str] = None,
        nano_model: Optional[str] = None,
        is_demo: bool = False
    ):
        self.provider = (provider or config.DEFAULT_PROVIDER).lower()
        self.is_demo = is_demo

        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = (
                config.NVIDIA_API_KEY if "nemotron" in self.provider else config.GROQ_API_KEY
            )

        if "nemotron" in self.provider or "nvidia" in self.provider:
            self.base_url = base_url or config.NVIDIA_BASE_URL
            self.ultra_model = ultra_model or config.NEMOTRON_ULTRA_MODEL
            self.nano_model = nano_model or config.NEMOTRON_NANO_MODEL
        else:
            # Default to Groq
            self.base_url = base_url or config.GROQ_BASE_URL
            self.ultra_model = ultra_model or config.GROQ_ULTRA_MODEL
            self.nano_model = nano_model or config.GROQ_NANO_MODEL

    async def call(
        self,
        messages: list,
        role: str = "ultra", # "ultra" for deep planning, "nano" for fast test-fixing
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        model = self.ultra_model if role == "ultra" else self.nano_model
        
        # If no API key configured: only allowed in explicitly labeled demo mode
        if not self.api_key or self.api_key.strip() == "" or self.api_key.startswith("demo_"):
            if self.is_demo:
                logger.info("Session labeled as demo. Using synthetic demo response.")
                simulated = self._simulate_response(messages, role)
                if simulated:
                    return simulated
            raise ValueError(f"No API key provided for {self.provider}. Cannot execute repository repair without a valid API key.")

        if "groq" in self.provider:
            # Respect Groq on-demand tier limits (1,000 output tokens/min)
            max_tokens = min(max_tokens, 700)

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json"
        }

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            payload["response_format"] = response_format

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(endpoint, headers=headers, json=payload)
                if resp.status_code in (413, 429):
                    # Rate limit or token overflow; extract wait time or sleep 2s and retry
                    wait_sec = 2.5
                    m = re.search(r"try again in ([0-9\.]+)s", resp.text)
                    if m:
                        wait_sec = float(m.group(1)) + 0.5
                    logger.warning(f"Groq token rate limit ({resp.status_code}); waiting {wait_sec:.1f}s before retry...")
                    await asyncio.sleep(wait_sec)
                    payload["max_tokens"] = min(payload.get("max_tokens", 700), 450)
                    resp = await client.post(endpoint, headers=headers, json=payload)

                if resp.status_code != 200:
                    err_msg = resp.text
                    logger.error(f"LLM API Error ({resp.status_code}): {err_msg}")
                    # Only fallback if explicitly labeled demo mode
                    if self.is_demo:
                        fallback = self._simulate_response(messages, role)
                        if fallback:
                            logger.info("Using graceful simulation fallback for demo run.")
                            return fallback
                    raise RuntimeError(f"{self.provider.upper()} API Error {resp.status_code}: {err_msg}")
                
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content
            except httpx.RequestError as e:
                logger.error(f"HTTP request error to {endpoint}: {e}")
                if self.is_demo:
                    fallback = self._simulate_response(messages, role)
                    if fallback:
                        return fallback
                raise

    def _simulate_response(self, messages: list, role: str) -> Optional[str]:
        """Provides high-quality synthetic responses strictly for built-in testbed demo repos when in demo mode."""
        text_context = " ".join([m.get("content", "") for m in messages]).lower()
        
        # Demo 2: UserAuth / JWT expiration leeway
        if "auth_validator" in text_context or "jwt" in text_context or "leeway" in text_context or "test_auth" in text_context:
            if role == "ultra":
                return json.dumps({
                    "issue_summary": "JWT Token verification fails due to subtracting leeway instead of adding it when testing current_time >= exp.",
                    "root_cause": "In auth_validator.py, the check 'current_time >= (exp - leeway_seconds)' prematurely expires tokens within the allowable leeway window.",
                    "suspected_files": ["auth_validator.py"],
                    "strategy": "Change comparison to 'current_time >= (exp + leeway_seconds)' to allow valid tokens within leeway window.",
                    "test_targets": ["tests/test_auth.py"]
                })
            else:
                return r"""```python:auth_validator.py
<<<<<<< SEARCH
    if current_time >= (exp - leeway_seconds):
        return False
=======
    if current_time >= (exp + leeway_seconds):
        return False
>>>>>>> REPLACE
```"""

        # Demo 3: URL Slug Generator
        elif "slugify" in text_context or "test_slug" in text_context or "slug" in text_context:
            if role == "ultra":
                return json.dumps({
                    "issue_summary": "Slug generator fails on non-ASCII characters and creates trailing/duplicate hyphens.",
                    "root_cause": "In slugify.py, the regex only allows [a-zA-Z0-9] and strips accented/unicode characters without normalization, leaving empty slugs.",
                    "suspected_files": ["slugify.py"],
                    "strategy": "Use unicodedata.normalize('NFKD') to decompose accented characters into ASCII counterparts and clean duplicate hyphens.",
                    "test_targets": ["tests/test_slug.py"]
                })
            else:
                return """```python:slugify.py
<<<<<<< SEARCH
def generate_slug(text: str) -> str:
    \"\"\"Converts a string into an SEO-friendly URL slug.\"\"\"
    if not text:
        return ""

    # BUG: Does not normalize accents with unicodedata NFKD before regex
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\\s-]", "", text)
    text = re.sub(r"[\\s-]+", "-", text)
    return text.strip("-")
=======
import unicodedata

def generate_slug(text: str) -> str:
    \"\"\"Converts a string into an SEO-friendly URL slug.\"\"\"
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\\s-]", "", text)
    text = re.sub(r"[\\s-]+", "-", text)
    return text.strip("-")
>>>>>>> REPLACE
```"""

        # Demo 1: StringCalc / negative numbers & custom delimiter
        elif "string_calculator" in text_context or "test_calculator" in text_context or "calculator" in text_context or "negatives not allowed" in text_context:
            if role == "ultra":
                return json.dumps({
                    "issue_summary": "StringCalculator raises ValueError when encountering negative numbers without listing all negatives, and fails on custom multi-character delimiters.",
                    "root_cause": "The add() function in string_calculator.py only catches the first negative number instead of collecting all negative numbers in the error message 'negatives not allowed: X, Y'. Also delimiter regex doesn't escape custom delimiters properly.",
                    "suspected_files": ["string_calculator.py"],
                    "strategy": "Collect all negative numbers in a list, format comma-separated error message, and refine delimiter splitting regex.",
                    "test_targets": ["tests/test_calculator.py"]
                })
            else: # nano fixer
                return r"""```python:string_calculator.py
<<<<<<< SEARCH
    # BUG: Only checks first negative number instead of collecting all
    for n in nums:
        if n < 0:
            raise ValueError(f"negatives not allowed: {n}")
=======
    negatives = [n for n in nums if n < 0]
    if negatives:
        neg_str = ", ".join(str(n) for n in negatives)
        raise ValueError(f"negatives not allowed: {neg_str}")
>>>>>>> REPLACE
```"""

        return None
