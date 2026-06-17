"""Dataclasses for avise/pipelines/imagegenerator/pipeline.py

Image-generator specific input/output contracts for the 4-phase pipeline.
EvaluationResult and ReportData are shared with the language-model pipeline
because they are already format-agnostic.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ImageGenSETCase:
    """Contract: Output of initialize(), input to execute().

    ID and prompt are required. Additional fields go to 'metadata'.
    """

    id: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            **self.metadata,
        }


@dataclass
class ImageGenExecutionOutput:
    """Single image generation execution result.

    Produced by execute() for each test case.
    'image_data' holds the raw image bytes when generation succeeded.
    'refused' is True when the API explicitly blocked the request.
    'response_text' holds any text returned alongside (or instead of) the
    image, e.g. a description, caption, or refusal explanation in prose.
    Some generators (e.g. OpenAI Responses API) return both an image and
    explanatory text; others (SD/Forge) return only an image.
    """

    set_id: str
    prompt: str
    image_data: Optional[bytes] = None  # None when generation was refused / errored
    refused: bool = False               # True when the API returned an explicit refusal
    response_text: Optional[str] = None # Text returned alongside or instead of the image
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "set_id": self.set_id,
            "prompt": self.prompt,
            "image_generated": self.image_data is not None,
            "refused": self.refused,
            "metadata": self.metadata,
        }
        if self.response_text:
            result["response_text"] = self.response_text
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class ImageGenOutputData:
    """Output of execute(), input to evaluate().

    Contains all execution outputs and the wall-clock duration.
    """

    outputs: List[ImageGenExecutionOutput]
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outputs": [o.to_dict() for o in self.outputs],
            "duration_seconds": self.duration_seconds,
        }
