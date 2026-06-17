"""Base class for image generator evaluators.

Image generator evaluators inspect the output of an image generation
request and determine whether policy was followed, violated, or the
result is inconclusive.

Unlike language-model evaluators (which do regex matching on text),
image evaluators work on image bytes and/or metadata from the API
response.  Future evaluators that use a vision model can extend this
base class and inspect the actual pixel content.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseImageGenEvaluator(ABC):
    """Abstract base class for image generator evaluators.

    Attributes:
        name: Unique identifier for the evaluator.
        description: What the evaluator detects.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def detect(
        self,
        image_data: Optional[bytes],
        metadata: dict,
    ) -> Tuple[bool, List[str]]:
        """Evaluate a single image generation output.

        Args:
            image_data: Raw image bytes returned by the connector, or None
                        when the request was refused / errored.
            metadata:   The ImageGenExecutionOutput metadata dict (includes
                        "refused", "error", and any SET-case metadata fields).

        Returns:
            Tuple:
                - bool: True if the evaluator's condition was detected.
                - List[str]: Human-readable description(s) of what was found.
        """
        pass
