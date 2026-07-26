"""Connector for OpenRouter (https://openrouter.ai) image generation models.

Unlike the OpenRouter language-model connector (which uses the OpenAI-SDK
compatible ``/chat/completions`` endpoint), image generation on OpenRouter is
served from a dedicated REST endpoint that isn't part of the OpenAI SDK's
image API surface, so this connector talks to it directly with ``requests``
(the same approach ``StableDiffusionConnector`` uses for its non-SDK backend):

    POST {api_url}/images         – text-to-image generation
    GET  {api_url}/images/models  – list of image-capable models (status check)

The images endpoint returns generated images as base64-encoded JSON
(``data: [{"b64_json": ..., "media_type": ...}]``), not raw bytes or a URL.

Connector config JSON shape
----------------------------
{
    "target_model": {
        "connector":    "openrouter-image",
        "type":         "image_generator",
        "name":         "bytedance-seed/seedream-4.5",
        "api_key":      "<your-openrouter-api-key>",
        "api_url":      null,
        "site_url":     null,
        "site_name":    null,
        "headers":      null,
        "n":            1,
        "aspect_ratio": null,
        "resolution":   null,
        "quality":      null,
        "output_format": null,
        "seed":         null,
        "timeout":      120,
        "generation_kwargs": {}
    }
}
"""

import base64
import logging
from typing import List, Optional

import requests

from .base import BaseImageGenConnector
from ...registry import connector_registry
from ...utils import ansi_colors

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

_IMAGES_PATH = "/images"
_IMAGE_MODELS_PATH = "/images/models"

# HTTP statuses OpenRouter uses for content-policy / guardrail blocks.
_REFUSAL_STATUS_CODES = (400, 403)
_REFUSAL_ERROR_TYPES = {"content_policy_violation", "refusal"}
_REFUSAL_KEYWORDS = (
    "safety",
    "nsfw",
    "blocked",
    "content policy",
    "moderat",
    "refused",
    "prohibited",
)


@connector_registry.register("openrouter-image")
class OpenRouterImageConnector(BaseImageGenConnector):
    """Connector for image generation models hosted on OpenRouter.

    Sends text prompts to OpenRouter's dedicated images endpoint and returns
    the decoded PNG/JPEG bytes (or None when the request is refused).

    Requires an OpenRouter API key, configured via the connector
    configuration file (api_key field).
    """

    name = "openrouter-image"

    def __init__(self, config: dict, evaluation: bool = False) -> None:
        """Initialise the OpenRouter image connector.

        Args:
            config: Dictionary from the connector configuration JSON.
            evaluation: Ignored for image generators (no eval model path).
                Accepted for API compatibility with the engine.

        Raises:
            KeyError: If "target_model" or a required field is missing.
        """
        if "target_model" not in config:
            raise KeyError(
                'OpenRouter Image Connector configuration file requires a '
                '"target_model" field. Refer to Connector documentations on '
                'how to configure connectors.'
            )
        target = config["target_model"]

        if not target.get("name"):
            raise KeyError(
                'OpenRouter image connector requires a model name. Add '
                '"target_model": {"name"} to connector configuration file as '
                'a string, e.g. "bytedance-seed/seedream-4.5".'
            )
        if not target.get("api_key"):
            raise KeyError(
                "OpenRouter image connector requires an API key. Add 'api_key' "
                "to connector configuration file as a string."
            )

        self.model: str = target["name"]
        self.api_key: str = target["api_key"]
        self.base_url: str = (target.get("api_url") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout: int = int(target.get("timeout") or 120)

        # Optional generation defaults (all overridable per-request via data dict)
        self.n: Optional[int] = target.get("n", 1)
        self.aspect_ratio: Optional[str] = target.get("aspect_ratio")
        self.resolution: Optional[str] = target.get("resolution")
        self.quality: Optional[str] = target.get("quality")
        self.output_format: Optional[str] = target.get("output_format")
        self.seed: Optional[int] = target.get("seed")
        self.generation_kwargs: dict = target.get("generation_kwargs") or {}

        headers = {"Authorization": f"Bearer {self.api_key}"}
        if target.get("site_url"):
            headers["HTTP-Referer"] = target["site_url"]
        if target.get("site_name"):
            headers["X-OpenRouter-Title"] = target["site_name"]
        if target.get("headers"):
            headers.update(target["headers"])

        self.session = requests.Session()
        self.session.headers.update(headers)

        logger.info("  OpenRouter Image Connector initialised")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  Model: {self.model}")
        logger.info(
            f"  API Key: {'*' * 8}...{self.api_key[-4:] if len(self.api_key) > 4 else '****'}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> dict:
        """Send a text prompt to the OpenRouter images API and return image bytes.

        Args:
            data: Generation parameters.
                Required:
                    - "prompt" (str): The text prompt to generate from.
                Optional (override connector defaults):
                    - "model" (str)
                    - "n" (int): Number of images to request. Only the
                      first returned image is used; the rest are discarded.
                    - "aspect_ratio" (str)
                    - "resolution" (str)
                    - "quality" (str)
                    - "output_format" (str)
                    - "seed" (int)

        Returns:
            dict:
                - "image_data" (bytes | None): Decoded image bytes, or None
                  when the request was refused or errored.
                - "refused" (bool): True when OpenRouter or the underlying
                  provider explicitly blocked the request (content policy
                  violation / guardrail), or silently returned no images.
                - "response_text" (str | None): Error/refusal message text
                  when present, otherwise None.

        Raises:
            KeyError: If "prompt" is missing from data.
            RuntimeError: If the API call fails unexpectedly.
        """
        if "prompt" not in data:
            raise KeyError(
                '"prompt" key is required in data dict for OpenRouterImageConnector.generate()'
            )

        payload = {
            "model": data.get("model", self.model),
            "prompt": data["prompt"],
        }
        for key, default in (
            ("n", self.n),
            ("aspect_ratio", self.aspect_ratio),
            ("resolution", self.resolution),
            ("quality", self.quality),
            ("output_format", self.output_format),
            ("seed", self.seed),
        ):
            value = data.get(key, default)
            if value is not None:
                payload[key] = value
        payload.update(self.generation_kwargs)

        url = self.base_url + _IMAGES_PATH
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.error(
                f"{ansi_colors['red']}OpenRouter images API request failed: {e}{ansi_colors['reset']}"
            )
            raise RuntimeError(f"Failed to reach OpenRouter images API: {e}") from e

        body = self._safe_json(response)

        if response.status_code != 200:
            if self._is_refusal(response.status_code, body):
                message = self._error_message(body)
                logger.info(
                    f"{ansi_colors['yellow']}OpenRouter images API refused the "
                    f"request (HTTP {response.status_code}): {message}{ansi_colors['reset']}"
                )
                return {"image_data": None, "refused": True, "response_text": message}
            raise RuntimeError(
                f"OpenRouter images API returned HTTP {response.status_code}: {body}"
            )

        images: list = body.get("data", []) if isinstance(body, dict) else []
        if not images:
            # API returned 200 but no images — treat as implicit refusal,
            # matching the convention used by the other image connectors.
            logger.info(
                f"{ansi_colors['yellow']}OpenRouter images API returned no images "
                f"(implicit refusal / safety filter).{ansi_colors['reset']}"
            )
            return {"image_data": None, "refused": True, "response_text": None}

        b64_image = images[0].get("b64_json")
        if not b64_image:
            raise RuntimeError(
                "OpenRouter images API returned an image entry without 'b64_json'."
            )

        try:
            image_bytes = base64.b64decode(b64_image)
        except Exception as e:
            raise RuntimeError(
                f"Failed to decode image from OpenRouter response: {e}"
            ) from e

        logger.debug(f"Image generated successfully ({len(image_bytes):,} bytes).")
        return {"image_data": image_bytes, "refused": False, "response_text": None}

    def status_check(self) -> bool:
        """Check that the OpenRouter images API is reachable.

        Returns:
            True if the API is reachable. Warns (but does not fail) if the
            configured model isn't found in the image-model catalog, since
            OpenRouter's catalog changes frequently.

        Raises:
            ConnectionError: If the API is not reachable.
        """
        try:
            model_ids = self._list_image_models()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to OpenRouter images API at {self.base_url}: {e}"
            ) from e

        logger.info(f"Available image models found: {len(model_ids)} models")

        if self.model in model_ids:
            logger.info(f"Model '{self.model}' is available.")
            return True

        logger.warning(
            f"Model '{self.model}' not found in available image models list. "
            f"Proceeding anyway as some models may not be listed."
        )
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_image_models(self) -> List[str]:
        """Helper method, used by status_check() to verify model availability.

        Returns:
            List of image-capable model ids.
        """
        url = self.base_url + _IMAGE_MODELS_PATH
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        body = self._safe_json(response)
        return [m["id"] for m in body.get("data", []) if "id" in m]

    def _is_refusal(self, status_code: int, body) -> bool:
        """Determine whether an error response represents a content/safety refusal.

        Arguments:
            status_code: HTTP status code of the response.
            body: Parsed JSON body (or raw text if parsing failed).

        Returns:
            True if the error looks like a content-policy/guardrail block
            rather than an unrelated API failure.
        """
        if status_code not in _REFUSAL_STATUS_CODES:
            return False

        if isinstance(body, dict):
            error = body.get("error", {})
            error_type = str((error.get("metadata") or {}).get("error_type", "")).lower()
            if error_type in _REFUSAL_ERROR_TYPES:
                return True
            message = str(error.get("message", "")).lower()
        else:
            message = str(body).lower()

        return any(keyword in message for keyword in _REFUSAL_KEYWORDS)

    @staticmethod
    def _error_message(body) -> Optional[str]:
        """Extract a human-readable error message from a parsed response body."""
        if isinstance(body, dict):
            return body.get("error", {}).get("message")
        return str(body) if body else None

    @staticmethod
    def _safe_json(response: requests.Response):
        """Parse JSON from a response, falling back to raw text."""
        try:
            return response.json()
        except Exception:
            return response.text
