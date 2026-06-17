"""Connector for AUTOMATIC1111 / Forge Stable Diffusion WebUI REST API.

Targets the local AUTOMATIC1111/Forge WebUI API at:
    POST {api_url}/sdapi/v1/txt2img   – text-to-image generation
    GET  {api_url}/sdapi/v1/sd-models – status / model list check

Connector config JSON shape (connector config file)
----------------------------------------------------
{
    "target_model": {
        "connector":        "stable-diffusion",
        "type":             "image_generator",
        "name":             "stable-diffusion",
        "api_url":          "http://localhost:7860",
        "api_key":          null,
        "steps":            20,
        "width":            512,
        "height":           512,
        "negative_prompt":  ""
    }
}
"""

import base64
import logging
from typing import Optional

import requests

from .base import BaseImageGenConnector
from ...registry import connector_registry
from ...utils.ansi_color_codes import ansi_colors

logger = logging.getLogger(__name__)

# AUTOMATIC1111 / Forge API paths
_TXT2IMG_PATH = "/sdapi/v1/txt2img"
_MODELS_PATH = "/sdapi/v1/sd-models"


@connector_registry.register("stable-diffusion")
class StableDiffusionConnector(BaseImageGenConnector):
    """Connector for the AUTOMATIC1111 / Forge Stable Diffusion WebUI.

    Sends text prompts to the local WebUI REST API and returns
    the raw PNG image bytes (or None when the request is refused).
    """

    name = "stable-diffusion"

    def __init__(self, config: dict, evaluation: bool = False) -> None:
        """Initialise the Stable Diffusion connector.

        Args:
            config: Dictionary from the connector configuration JSON.
            evaluation: Ignored for image generators (no eval model path).
                Accepted for API compatibility with the engine.
        """
        # Image generator connectors only have a target_model entry
        target = config["target_model"]

        self.model: str = target.get("name", "stable-diffusion")
        self.base_url: str = target.get("api_url", "http://localhost:7860").rstrip("/")
        self.api_key: Optional[str] = target.get("api_key")

        # Generation defaults (all overridable per-request via data dict)
        self.steps: int = int(target.get("steps", 20))
        self.width: int = int(target.get("width", 512))
        self.height: int = int(target.get("height", 512))
        self.negative_prompt: str = target.get("negative_prompt", "")

        # Build a requests session with optional auth header
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        logger.info("  Stable Diffusion Connector initialised")
        logger.info(f"  Base URL : {self.base_url}")
        logger.info(f"  Model    : {self.model}")
        if self.api_key:
            logger.info(
                f"  API Key  : {'*' * 8}...{self.api_key[-4:] if len(self.api_key) > 4 else '****'}"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> dict:
        """Send a text prompt to the SD WebUI and return the image bytes.

        Args:
            data: Generation parameters.
                Required:
                    - "prompt" (str)
                Optional (override connector defaults):
                    - "negative_prompt" (str)
                    - "steps"           (int)
                    - "width"           (int)
                    - "height"          (int)
                    - "seed"            (int, -1 for random)

        Returns:
            dict:
                - "image_data" (bytes | None): PNG image bytes, or None
                  when the API refused/blocked the request.
                - "refused" (bool): True when the API returned an explicit
                  safety/NSFW block (HTTP 400/422 with a safety message,
                  or images list is empty due to filtering).

        Raises:
            KeyError: If "prompt" is missing from data.
            RuntimeError: If the API call fails unexpectedly.
        """
        if "prompt" not in data:
            raise KeyError(
                '"prompt" key is required in data dict for StableDiffusionConnector.generate()'
            )

        payload = {
            "prompt": data["prompt"],
            "negative_prompt": data.get("negative_prompt", self.negative_prompt),
            "steps": data.get("steps", self.steps),
            "width": data.get("width", self.width),
            "height": data.get("height", self.height),
            "seed": data.get("seed", -1),
            "batch_size": 1,
            "n_iter": 1,
            "save_images": False,
            "send_images": True,
        }

        url = self.base_url + _TXT2IMG_PATH

        try:
            response = self.session.post(url, json=payload, timeout=120)
        except requests.exceptions.RequestException as e:
            logger.error(
                f"{ansi_colors['red']}SD API request failed: {e}{ansi_colors['reset']}"
            )
            raise RuntimeError(f"Failed to reach Stable Diffusion API: {e}") from e

        # Detect explicit API-level refusals (safety filter blocks)
        if response.status_code in (400, 422):
            body = self._safe_json(response)
            msg = str(body).lower()
            if any(kw in msg for kw in ("safety", "nsfw", "blocked", "filter", "rejected")):
                logger.info(
                    f"{ansi_colors['yellow']}SD API refused the request (safety filter): "
                    f"{body}{ansi_colors['reset']}"
                )
                return {"image_data": None, "refused": True}
            # Non-safety HTTP error → propagate as RuntimeError
            raise RuntimeError(
                f"SD API returned HTTP {response.status_code}: {body}"
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"SD API returned unexpected HTTP {response.status_code}: "
                f"{self._safe_json(response)}"
            )

        body = self._safe_json(response)
        images: list = body.get("images", [])

        if not images:
            # API returned 200 but no images — treat as implicit refusal
            logger.info(
                f"{ansi_colors['yellow']}SD API returned no images (implicit refusal / safety filter).{ansi_colors['reset']}"
            )
            return {"image_data": None, "refused": True}

        # Decode the first image from base64 PNG
        try:
            image_bytes = base64.b64decode(images[0])
        except Exception as e:
            raise RuntimeError(f"Failed to decode image from SD API response: {e}") from e

        logger.debug(
            f"Image generated successfully ({len(image_bytes):,} bytes)."
        )
        return {"image_data": image_bytes, "refused": False}

    def status_check(self) -> bool:
        """Check that the AUTOMATIC1111/Forge API is reachable.

        Hits GET /sdapi/v1/sd-models to confirm the API is up.
        Does *not* verify a specific model is loaded, because SD WebUI
        loads a single active checkpoint, not a named model per request.

        Returns:
            True if the API is up.

        Raises:
            ConnectionError: If the API is not reachable.
        """
        url = self.base_url + _MODELS_PATH
        try:
            response = self.session.get(url, timeout=10)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Stable Diffusion API at {self.base_url}: {e}"
            ) from e

        if response.status_code == 200:
            logger.info(f"Stable Diffusion API at {self.base_url} is reachable.")
            return True

        raise ConnectionError(
            f"Stable Diffusion API at {self.base_url} returned HTTP "
            f"{response.status_code} during status check."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_json(response: requests.Response):
        """Parse JSON from a response, falling back to raw text."""
        try:
            return response.json()
        except Exception:
            return response.text
