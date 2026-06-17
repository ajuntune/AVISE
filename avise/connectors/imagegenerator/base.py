"""Base class for image generator connectors.

Image generator connectors handle communication with image generation
backends (Stable Diffusion, ComfyUI, cloud APIs, etc.), abstracting
the specifics of each API so that SET cases can be written once and
run against different backends.

Class contract
--------------
generate(data: dict) -> dict
    Send a prompt to the image generator and return a result dict:
        {
            "image_data":    bytes | None,  # raw image bytes, or None if refused / error
            "refused":       bool,          # True when the API explicitly blocked the request
            "response_text": str | None,    # optional: text returned alongside or instead of
                                            # the image (e.g. OpenAI caption / refusal prose).
                                            # Omit or set to None if the API returns no text.
        }

status_check() -> bool
    Verify that the target API endpoint is reachable and ready.
    Raises ConnectionError or ValueError on failure.

Required attributes
-------------------
name : str
    Human-readable connector name (also used as the 'model' identifier
    exposed to the pipeline's run() method).
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseImageGenConnector(ABC):
    """Abstract base class for image generator connectors.

    Each concrete connector targets a specific backend
    (e.g. AUTOMATIC1111/Forge, ComfyUI, Stability AI).
    """

    name: str = ""

    @abstractmethod
    def generate(self, data: dict) -> dict:
        """Send a prompt to the image generator and return the result.

        Args:
            data: Dictionary with generation parameters.
                Required keys:
                    - "prompt" (str): The text prompt to generate from.
                Optional keys (connector-specific):
                    - "negative_prompt" (str)
                    - "steps" (int)
                    - "width" (int)
                    - "height" (int)
                    - "seed" (int)

        Returns:
            dict with keys:
                - "image_data" (bytes | None): Raw image bytes, or None if
                  the request was refused or an error occurred.
                - "refused" (bool): True if the API explicitly blocked the
                  request (e.g. NSFW filter, safety system).

        Raises:
            RuntimeError: If the API call fails unexpectedly.
        """
        pass

    @abstractmethod
    def status_check(self) -> bool:
        """Verify that the target API is reachable.

        Returns:
            True if the API is reachable and ready.

        Raises:
            ConnectionError: If the API is not reachable.
            ValueError: If the configured model / checkpoint is not found.
        """
        pass
