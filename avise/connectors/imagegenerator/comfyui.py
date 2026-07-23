"""Connector for ComfyUI image generation API.

ComfyUI exposes a REST API that works in three steps:

1. ``POST /prompt``          – Queue a workflow; returns a ``prompt_id``.
2. ``GET  /history/{id}``    – Poll until the job finishes.
3. ``GET  /view?filename=…`` – Download the generated image bytes.

Unlike AUTOMATIC1111, ComfyUI is workflow-driven: generation is described
as a node graph (JSON).  This connector ships a sensible default KSampler
workflow for text-to-image generation, but you can supply any custom
workflow via ``workflow_path`` in the connector config.

Prompt injection
----------------
The connector finds the ``CLIPTextEncode`` node identified by
``prompt_node_id`` (default ``"2"``) in the workflow and replaces its
``text`` input with the prompt string from AVISE.  Likewise,
``negative_prompt_node_id`` (default ``"3"``) receives the
``negative_prompt``.

Custom workflows
----------------
Set ``workflow_path`` in the connector config to the absolute path of a
ComfyUI-format workflow JSON.  The connector will load it and inject the
prompt into the designated node before queuing.

Safety refusals
---------------
ComfyUI does not apply safety filtering by default.  If a safety-checker
node (e.g. ``NSFW_Detector``) is present in the workflow and blocks the
generation, the output images list will be empty.  The connector maps this
to ``refused = True, image_data = None``.

Connector config JSON shape
---------------------------
::

    {
        "target_model": {
            "connector":               "comfyui",
            "type":                    "image_generator",
            "name":                    "comfyui",
            "api_url":                 "http://localhost:8188",
            "checkpoint":              "v1-5-pruned-emaonly.safetensors",
            "width":                   512,
            "height":                  512,
            "steps":                   20,
            "cfg_scale":               7.0,
            "sampler_name":            "euler",
            "scheduler":               "normal",
            "negative_prompt":         "",
            "timeout":                 120,
            "poll_interval":           1.0,
            "prompt_node_id":          "2",
            "negative_prompt_node_id": "3",
            "workflow_path":           null
        }
    }

Requires the ``requests`` package (already a dependency of AVISE).
"""

import copy
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

from .base import BaseImageGenConnector
from ...registry import connector_registry
from ...utils.ansi_color_codes import ansi_colors

logger = logging.getLogger(__name__)

# ComfyUI API paths
_PROMPT_PATH    = "/prompt"
_HISTORY_PATH   = "/history/{prompt_id}"
_VIEW_PATH      = "/view"
_SYSTEM_STATS   = "/system_stats"

# ------------------------------------------------------------------
# Default KSampler workflow
# ------------------------------------------------------------------
# Node layout:
#   "1" CheckpointLoaderSimple  → model / clip / vae
#   "2" CLIPTextEncode (pos)    → conditioning (positive)
#   "3" CLIPTextEncode (neg)    → conditioning (negative)
#   "4" EmptyLatentImage        → latent tensor
#   "5" KSampler                → samples
#   "6" VAEDecode               → pixels
#   "7" SaveImage               → output image (connector reads this)
# ------------------------------------------------------------------
_DEFAULT_WORKFLOW: dict = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "__CHECKPOINT__",
        },
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "__POSITIVE_PROMPT__",
            "clip": ["1", 1],
        },
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "__NEGATIVE_PROMPT__",
            "clip": ["1", 1],
        },
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 1,
        },
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
        },
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["5", 0],
            "vae": ["1", 2],
        },
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "avise",
            "images": ["6", 0],
        },
    },
}


@connector_registry.register("comfyui")
class ComfyUIConnector(BaseImageGenConnector):
    """Connector for ComfyUI image generation.

    Queues a text-to-image workflow via the ComfyUI REST API, polls for
    completion, and returns the generated image as raw bytes.

    Works with the built-in default KSampler workflow or any custom
    workflow JSON supplied via ``workflow_path``.
    """

    name = "comfyui"

    # Subclasses can override this to prepend a path prefix.
    # ComfyUICloudConnector uses "/api" to target cloud.comfy.org.
    _PATH_PREFIX: str = ""

    def __init__(self, config: dict, evaluation: bool = False) -> None:
        """Initialise the ComfyUI connector.

        Args:
            config: Dictionary loaded from the connector configuration JSON.
                    Reads from ``config["target_model"]``.
            evaluation: Unused; present for registry API compatibility.
        """
        target = config.get("target_model", {})

        self.model: str              = target.get("name", "comfyui")
        self.base_url: str           = target.get("api_url", "http://localhost:8188").rstrip("/")
        self.checkpoint: str         = target.get("checkpoint", "v1-5-pruned-emaonly.safetensors")
        self.width: int              = int(target.get("width", 512))
        self.height: int             = int(target.get("height", 512))
        self.steps: int              = int(target.get("steps", 20))
        self.cfg_scale: float        = float(target.get("cfg_scale", 7.0))
        self.sampler_name: str       = target.get("sampler_name", "euler")
        self.scheduler: str          = target.get("scheduler", "normal")
        self.negative_prompt: str    = target.get("negative_prompt", "")
        self.timeout: float          = float(target.get("timeout", 120))
        self.poll_interval: float    = float(target.get("poll_interval", 1.0))
        self.prompt_node_id: str     = str(target.get("prompt_node_id", "2"))
        self.neg_prompt_node_id: str = str(target.get("negative_prompt_node_id", "3"))

        # Optional path to a custom workflow JSON file
        workflow_path_str: Optional[str] = target.get("workflow_path")
        self._custom_workflow: Optional[dict] = None
        if workflow_path_str:
            workflow_path = Path(workflow_path_str)
            if not workflow_path.exists():
                raise FileNotFoundError(
                    f"ComfyUI workflow file not found: {workflow_path}"
                )
            with open(workflow_path, "r", encoding="utf-8") as f:
                self._custom_workflow = json.load(f)
            logger.info(f"Loaded custom ComfyUI workflow from {workflow_path}")

        # Stable session with JSON content-type
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Unique client ID for this connector session (used by ComfyUI internally)
        self._client_id = str(uuid.uuid4())

        logger.info(
            f"{ansi_colors['green']}ComfyUIConnector initialised "
            f"(url: {self.base_url}, checkpoint: {self.checkpoint})"
            f"{ansi_colors['reset']}"
        )

    def _url(self, path: str) -> str:
        """Build a full URL by prepending base_url and the path prefix."""
        return self.base_url + self._PATH_PREFIX + path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> dict:
        """Queue a generation workflow and return the image bytes.

        Args:
            data: Generation parameters.
                Required:
                    - ``"prompt"`` (str): The positive text prompt.
                Optional:
                    - ``"negative_prompt"`` (str): Override negative prompt.
                    - ``"steps"``           (int)
                    - ``"width"``           (int)
                    - ``"height"``          (int)
                    - ``"cfg_scale"``       (float)
                    - ``"seed"``            (int, -1 or omit for random)

        Returns:
            dict with keys:
                - ``"image_data"`` (bytes | None): PNG/JPEG image bytes,
                  or None when the generation was blocked or produced no output.
                - ``"refused"`` (bool): True when no image was returned
                  (implicit safety block or workflow error).
                - ``"response_text"`` (str | None): Always None for ComfyUI
                  (no accompanying text output).

        Raises:
            KeyError: If ``"prompt"`` is missing from data.
            RuntimeError: If the API call fails or times out.
        """
        if "prompt" not in data:
            raise KeyError('"prompt" key is required in data dict for ComfyUIConnector.generate()')

        prompt       = data["prompt"]
        neg_prompt   = data.get("negative_prompt", self.negative_prompt)
        steps        = int(data.get("steps", self.steps))
        width        = int(data.get("width", self.width))
        height       = int(data.get("height", self.height))
        cfg_scale    = float(data.get("cfg_scale", self.cfg_scale))
        seed         = int(data.get("seed", int(uuid.uuid4().int % (2**32))))

        logger.info(
            f"{ansi_colors['cyan']}Sending prompt to ComfyUI: "
            f"{prompt[:120]!r}{'...' if len(prompt) > 120 else ''}"
            f"{ansi_colors['reset']}"
        )

        workflow = self._build_workflow(
            prompt=prompt,
            neg_prompt=neg_prompt,
            steps=steps,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
            seed=seed,
        )

        prompt_id = self._queue_prompt(workflow)
        logger.info(f"Workflow queued (prompt_id: {prompt_id})")

        output = self._wait_for_completion(prompt_id)
        if output is None:
            logger.info(
                f"{ansi_colors['yellow']}ComfyUI returned no image output "
                f"(possible safety block or workflow error).{ansi_colors['reset']}"
            )
            return {"image_data": None, "refused": True, "response_text": None}

        image_bytes = self._download_image(output)
        logger.info(
            f"{ansi_colors['green']}Image generated successfully "
            f"({len(image_bytes):,} bytes){ansi_colors['reset']}"
        )
        return {"image_data": image_bytes, "refused": False, "response_text": None}

    def status_check(self) -> bool:
        """Verify that the ComfyUI API is reachable.

        Hits ``GET /system_stats`` which returns system info without
        triggering any generation.

        Returns:
            True if the API is reachable.

        Raises:
            ConnectionError: If the API is not reachable.
        """
        url = self._url(_SYSTEM_STATS)
        try:
            response = self.session.get(url, timeout=10)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to ComfyUI API at {self.base_url}: {e}"
            ) from e

        if response.status_code == 200:
            logger.info(f"ComfyUI API at {self.base_url} is reachable.")
            return True

        raise ConnectionError(
            f"ComfyUI API at {self.base_url} returned HTTP "
            f"{response.status_code} during status check."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_workflow(
        self,
        prompt: str,
        neg_prompt: str,
        steps: int,
        width: int,
        height: int,
        cfg_scale: float,
        seed: int,
    ) -> dict:
        """Build a complete ComfyUI workflow dict ready to be queued.

        Uses the custom workflow if one was loaded from ``workflow_path``,
        otherwise falls back to the built-in default KSampler workflow.
        In both cases, the text prompt is injected into the designated node.

        Args:
            prompt:     Positive text prompt.
            neg_prompt: Negative text prompt.
            steps:      Number of sampling steps.
            width:      Image width in pixels.
            height:     Image height in pixels.
            cfg_scale:  Classifier-free guidance scale.
            seed:       Random seed for reproducibility.

        Returns:
            ComfyUI workflow dict with prompt injected.
        """
        if self._custom_workflow is not None:
            workflow = copy.deepcopy(self._custom_workflow)
            # ComfyUI workflow exports can contain top-level metadata fields
            # (e.g. "version": 3) that are not nodes.  ComfyUI's
            # node_replace_manager checks "class_type" in node_struct, which
            # raises TypeError when node_struct is an int/float/str.  Strip
            # any non-dict entries before sending.
            non_nodes = [k for k, v in workflow.items() if not isinstance(v, dict)]
            if non_nodes:
                logger.debug(
                    f"Stripping {len(non_nodes)} non-node top-level key(s) from "
                    f"custom workflow before queuing: {non_nodes}"
                )
                for k in non_nodes:
                    del workflow[k]
        else:
            workflow = copy.deepcopy(_DEFAULT_WORKFLOW)
            # Fill in the default workflow's configurable fields
            workflow["1"]["inputs"]["ckpt_name"]   = self.checkpoint
            workflow["4"]["inputs"]["width"]        = width
            workflow["4"]["inputs"]["height"]       = height
            workflow["5"]["inputs"]["steps"]        = steps
            workflow["5"]["inputs"]["cfg"]          = cfg_scale
            workflow["5"]["inputs"]["sampler_name"] = self.sampler_name
            workflow["5"]["inputs"]["scheduler"]    = self.scheduler
            workflow["5"]["inputs"]["seed"]         = seed

        # Inject the prompts into the designated CLIPTextEncode nodes
        if self.prompt_node_id in workflow:
            workflow[self.prompt_node_id]["inputs"]["text"] = prompt
        else:
            logger.warning(
                f"prompt_node_id '{self.prompt_node_id}' not found in workflow. "
                f"Prompt will not be injected."
            )

        if self.neg_prompt_node_id in workflow:
            workflow[self.neg_prompt_node_id]["inputs"]["text"] = neg_prompt
        else:
            logger.debug(
                f"negative_prompt_node_id '{self.neg_prompt_node_id}' not found "
                f"in workflow — skipping negative prompt injection."
            )

        return workflow

    def _queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow to the ComfyUI queue and return the prompt_id.

        Args:
            workflow: ComfyUI workflow dict.

        Returns:
            The ``prompt_id`` string assigned by ComfyUI.

        Raises:
            RuntimeError: If the API returns a non-200 status or node errors.
        """
        payload = {"prompt": workflow, "client_id": self._client_id}
        url = self._url(_PROMPT_PATH)

        try:
            response = self.session.post(url, json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to connect to ComfyUI API at {url}: {e}"
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"ComfyUI /prompt returned HTTP {response.status_code}: "
                f"{self._safe_json(response)}"
            )

        body = self._safe_json(response)

        # Warn about node validation errors (workflow misconfiguration)
        node_errors = body.get("node_errors", {})
        if node_errors:
            logger.warning(
                f"ComfyUI reported node errors in workflow: {node_errors}. "
                f"Generation may fail."
            )

        prompt_id: Optional[str] = body.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(
                f"ComfyUI /prompt response missing 'prompt_id': {body}"
            )

        return prompt_id

    def _wait_for_completion(self, prompt_id: str) -> Optional[dict]:
        """Poll ``/history/{prompt_id}`` until the job is done or times out.

        Args:
            prompt_id: The ID returned by ``_queue_prompt()``.

        Returns:
            The image info dict from the first SaveImage output node, or
            None if no image was produced.

        Raises:
            RuntimeError: If the job fails or times out.
        """
        url = (self.base_url + self._PATH_PREFIX + _HISTORY_PATH).format(prompt_id=prompt_id)
        deadline = time.monotonic() + self.timeout

        while time.monotonic() < deadline:
            try:
                response = self.session.get(url, timeout=10)
            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"ComfyUI /history request failed: {e}"
                ) from e

            if response.status_code != 200:
                raise RuntimeError(
                    f"ComfyUI /history returned HTTP {response.status_code}"
                )

            history = self._safe_json(response)

            if prompt_id not in history:
                # Job not finished yet — keep polling
                time.sleep(self.poll_interval)
                continue

            job = history[prompt_id]
            status_info = job.get("status", {})

            if not status_info.get("completed", False):
                time.sleep(self.poll_interval)
                continue

            status_string = status_info.get("status_string", "")
            if status_string == "error":
                messages = status_info.get("messages", [])
                raise RuntimeError(
                    f"ComfyUI job {prompt_id} failed: {messages}"
                )

            # Walk all output nodes to find the first image
            outputs: dict = job.get("outputs", {})
            for node_id, node_output in outputs.items():
                images: list = node_output.get("images", [])
                if images:
                    image_info = images[0]
                    logger.debug(
                        f"Found image in output node '{node_id}': {image_info}"
                    )
                    return image_info

            # Job completed but no images produced
            return None

        raise RuntimeError(
            f"ComfyUI job {prompt_id} timed out after {self.timeout:.0f}s."
        )

    def _download_image(self, image_info: dict) -> bytes:
        """Download image bytes from ComfyUI's ``/view`` endpoint.

        Args:
            image_info: Dict with ``filename``, ``type``, and ``subfolder``
                        keys, as returned by the history API.

        Returns:
            Raw image bytes.

        Raises:
            RuntimeError: If the download fails.
        """
        params = {
            "filename": image_info["filename"],
            "type":     image_info.get("type", "output"),
            "subfolder": image_info.get("subfolder", ""),
        }
        url = self._url(_VIEW_PATH)

        try:
            response = self.session.get(url, params=params, timeout=60)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to download image from ComfyUI: {e}"
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"ComfyUI /view returned HTTP {response.status_code} "
                f"for {params}"
            )

        return response.content

    @staticmethod
    def _safe_json(response: requests.Response):
        """Parse JSON from a response, falling back to raw text on failure."""
        try:
            return response.json()
        except Exception:
            return response.text
