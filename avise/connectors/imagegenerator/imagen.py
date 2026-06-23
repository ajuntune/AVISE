"""Connector for Google Imagen via Vertex AI.

Uses the ``generate_images()`` SDK method, which is distinct from the
``generate_content()`` method used by the Gemini image models (gemini-*-image).

Imagen returns **only image bytes** — there are no accompanying text parts.

Authentication uses Google Cloud Application Default Credentials (ADC).
Before running, authenticate with::

    gcloud auth application-default login

Or point to a service account key file::

    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

Available models (Imagen 4)
---------------------------
    imagen-4.0-generate-001         – Standard quality (recommended default)
    imagen-4.0-ultra-generate-001   – Highest quality, slower
    imagen-4.0-fast-generate-001    – Fastest, lower quality

Connector config JSON shape
---------------------------
{
    "target_model": {
        "connector":        "imagen",
        "type":             "image_generator",
        "name":             "imagen-4.0-generate-001",
        "project_id":       "<your-gcp-project-id>",
        "location":         "us-central1",
        "model":            "imagen-4.0-generate-001",
        "aspect_ratio":     "1:1",
        "number_of_images": 1
    }
}
"""

import logging

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
#except ImportError:
#    _GENAI_AVAILABLE = False
except Exception as e:
    import traceback
    traceback.print_exc()
    raise

from .base import BaseImageGenConnector
from ...registry import connector_registry
from ...utils.ansi_color_codes import ansi_colors

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "imagen-4.0-generate-001"


@connector_registry.register("imagen")
class ImagenConnector(BaseImageGenConnector):
    """Connector for Google Imagen image generation models via Vertex AI.

    Uses the ``generate_images()`` endpoint, returning image bytes directly
    with no accompanying text. Authentication via Application Default
    Credentials (ADC) — run ``gcloud auth application-default login`` first,
    or set ``GOOGLE_APPLICATION_CREDENTIALS`` to a service account key file.

    Requires the ``google-genai`` package:
        pip install google-genai
    """

    name = "imagen"

    def __init__(self, config: dict, evaluation: bool = False) -> None:
        """Initialise the Imagen connector.

        Args:
            config: Dictionary from the connector configuration JSON.
            evaluation: Ignored for image generators. Accepted for API
                compatibility with the engine.

        Raises:
            ImportError: If the ``google-genai`` package is not installed.
            KeyError: If ``project_id`` is missing from the config.
        """
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "The 'google-genai' package is required for ImagenConnector. "
                "Install it with:  pip install google-genai"
            )

        target = config["target_model"]

        self.model: str = target.get("model", _DEFAULT_MODEL)
        self.name = self.model
        self.aspect_ratio: str = target.get("aspect_ratio", "1:1")
        self.number_of_images: int = int(target.get("number_of_images", 1))

        project_id: str = target.get("project_id", "")
        location: str = target.get("location", "us-central1")

        if not project_id:
            raise KeyError(
                "ImagenConnector requires 'project_id' in the connector config."
            )

        self._client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

        logger.info("  Imagen Connector initialised (Vertex AI)")
        logger.info(f"  Project      : {project_id}")
        logger.info(f"  Location     : {location}")
        logger.info(f"  Model        : {self.model}")
        logger.info(f"  Aspect ratio : {self.aspect_ratio}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> dict:
        """Send a text prompt to the Imagen API and return image bytes.

        Args:
            data: Generation parameters.
                Required:
                    - ``"prompt"`` (str): Text prompt to generate from.
                Optional:
                    - ``"model"`` (str): Override the connector's default model.
                    - ``"aspect_ratio"`` (str): e.g. ``"1:1"``, ``"16:9"``,
                      ``"4:3"``, ``"3:4"``, ``"9:16"``.
                    - ``"number_of_images"`` (int): 1–4. Only the first image
                      is returned; the rest are discarded.

        Returns:
            dict:
                - ``"image_data"`` (bytes | None): Raw PNG image bytes,
                  or ``None`` when the request was refused or errored.
                - ``"refused"`` (bool): ``True`` when the safety system
                  explicitly blocked the request.
                - ``"response_text"`` (None): Always ``None`` — Imagen
                  returns images only, no text.

        Raises:
            KeyError: If ``"prompt"`` is missing from ``data``.
            RuntimeError: If the API call fails unexpectedly.
        """
        if "prompt" not in data:
            raise KeyError(
                '"prompt" key is required in data dict for ImagenConnector.generate()'
            )

        prompt = data["prompt"]
        model = data.get("model", self.model)
        aspect_ratio = data.get("aspect_ratio", self.aspect_ratio)
        number_of_images = int(data.get("number_of_images", self.number_of_images))

        try:
            response = self._client.models.generate_images(
                model=model,
                prompt=prompt,
                config=genai_types.GenerateImagesConfig(
                    number_of_images=number_of_images,
                    aspect_ratio=aspect_ratio,
                ),
            )

            # Accessing response.generated_images can itself raise TypeError
            # when the SDK processes a safety-blocked payload that omits
            # fields the SDK expects (e.g. len(None)).
            generated = response.generated_images or []
            if not generated:
                logger.info(
                    f"{ansi_colors['yellow']}Imagen returned no images "
                    f"(implicit refusal / safety block).{ansi_colors['reset']}"
                )
                return {"image_data": None, "refused": True, "response_text": None}

            image_bytes: bytes = generated[0].image.image_bytes
            if not image_bytes:
                logger.info(
                    f"{ansi_colors['yellow']}Imagen returned an image entry with no bytes "
                    f"(safety block).{ansi_colors['reset']}"
                )
                return {"image_data": None, "refused": True, "response_text": None}

        except TypeError as e:
            # The Vertex AI SDK raises TypeError when it tries to process a
            # safety-blocked response whose payload omits expected fields
            # (e.g. "object of type 'NoneType' has no len()").
            # Treat this as a refusal rather than an unexpected error.
            logger.info(
                f"{ansi_colors['yellow']}Imagen safety block detected "
                f"(SDK TypeError - response payload indicates refusal): "
                f"{e}{ansi_colors['reset']}"
            )
            return {"image_data": None, "refused": True, "response_text": None}
        except Exception as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ("safety", "blocked", "policy", "harm", "prohibited")):
                logger.info(
                    f"{ansi_colors['yellow']}Imagen API refused the request "
                    f"(safety/policy): {e}{ansi_colors['reset']}"
                )
                return {"image_data": None, "refused": True, "response_text": None}
            raise RuntimeError(f"Imagen API call failed: {e}") from e

        logger.debug(f"Imagen image generated ({len(image_bytes):,} bytes).")
        return {"image_data": image_bytes, "refused": False, "response_text": None}

    def status_check(self) -> bool:
        """Verify that the Vertex AI API is reachable with the configured credentials.

        Returns:
            True if the API is reachable and credentials are valid.

        Raises:
            ConnectionError: If the API is not reachable or credentials are invalid.
        """
        try:
            models = list(self._client.models.list())
            if models:
                logger.info(f"Imagen Vertex AI API reachable. ({len(models)} models available)")
                return True
            raise ConnectionError("Vertex AI returned an empty model list.")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Imagen Vertex AI API: {e}") from e
