"""Connector for OpenRouter (https://openrouter.ai) API communication.

OpenRouter exposes a single OpenAI-compatible Chat Completions endpoint that
routes to many different underlying providers/models (e.g. "openai/gpt-4o-mini",
"anthropic/claude-3.7-sonnet", "google/gemini-2.5-flash"). This connector uses
the OpenAI Python SDK pointed at OpenRouter's endpoint, since OpenRouter is a
documented drop-in replacement for it — but calls the Chat Completions API
(`chat.completions.create`), not the Responses API, since OpenRouter only
supports the former.
"""

import logging
from typing import List, Optional

from openai import OpenAI

from .base import BaseLMConnector, Message
from ...registry import connector_registry
from ...utils import ansi_colors

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@connector_registry.register("openrouter-lm")
class OpenRouterLMConnector(BaseLMConnector):
    """Connector for communicating with the OpenRouter API.

    Used by SETs for sending prompts to any model available on OpenRouter
    and collecting their responses. Supports both simple generation and
    generation with system prompts.

    Requires an OpenRouter API key, configured via the connector
    configuration file (api_key field).
    """

    name = "openrouter-lm"

    def __init__(self, config: dict, evaluation: bool = False):
        """Initialize the OpenRouter connector.

        Args:
            config: Connector configuration data.
            evaluation: Whether this connector instance is for the
                evaluation model ("eval_model") or the target model
                ("target_model").

        Raises:
            KeyError: If required fields are missing from configuration data.
            TypeError: If configuration data is of a wrong type.
            SystemError: If failed to initialize the OpenAI client.
        """
        section_key = "eval_model" if evaluation else "target_model"
        model_config = self._parse_model_config(config, section_key)

        self.model = model_config["name"]
        self.api_key = model_config["api_key"]
        self.base_url = model_config["api_url"]
        self.max_tokens = model_config["max_tokens"]
        self.completion_kwargs = model_config["completion_kwargs"]

        headers = {}
        if model_config["site_url"]:
            headers["HTTP-Referer"] = model_config["site_url"]
        if model_config["site_name"]:
            headers["X-OpenRouter-Title"] = model_config["site_name"]
        if model_config["headers"]:
            headers.update(model_config["headers"])
        self.headers = headers or None

        try:
            client_kwargs = {"api_key": self.api_key, "base_url": self.base_url}
            if self.headers is not None:
                client_kwargs["default_headers"] = self.headers

            self.client = OpenAI(**client_kwargs)
        except Exception as e:
            logger.error("Failed to initialize OpenRouter client.")
            raise SystemError from e

        logger.info(f"  OpenRouter Connector Initialized")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(
            f"  API Key: {'*' * 8}...{self.api_key[-4:] if len(self.api_key) > 4 else '****'}"
        )

    def _parse_model_config(self, config: dict, section_key: str) -> dict:
        """Validate and extract the model configuration for one section.

        Arguments:
            config: Full connector configuration data.
            section_key: Either "target_model" or "eval_model".

        Returns:
            Dict with the validated/defaulted fields for this section.

        Raises:
            KeyError: If a required field is missing.
            TypeError: If a field has the wrong type.
        """
        if section_key not in config:
            raise KeyError(
                f'OpenRouter Connector configuration file requires a "{section_key}" '
                f"field. Refer to Connector documentations on how to configure connectors."
            )
        section = config[section_key]

        if "name" not in section:
            raise KeyError(
                f'OpenRouter connector requires a model name. Add "{section_key}": '
                f'{{"name"}} to connector configuration file as a string, e.g. '
                f'"openai/gpt-4o-mini".'
            )
        if not isinstance(section["name"], str):
            raise TypeError(
                f'OpenRouter connector requires a model "name" for the {section_key} '
                f"as a STRING."
            )

        if "api_key" not in section:
            raise KeyError(
                f"OpenRouter Connector requires an API key for the {section_key}. "
                f"Add 'api_key' to connector configuration file as a string."
            )
        if not isinstance(section["api_key"], str):
            raise TypeError(
                f"OpenRouter connector requires an API key for the {section_key} as a STRING."
            )

        api_url = section.get("api_url") or DEFAULT_BASE_URL
        if not isinstance(api_url, str):
            raise TypeError(
                f'OpenRouter connector requires an API URL for the {section_key} as a '
                f"STRING or null (defaults to {DEFAULT_BASE_URL})."
            )

        for optional_str_field in ("site_url", "site_name"):
            value = section.get(optional_str_field)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f'OpenRouter connector "{optional_str_field}" for the {section_key} '
                    f"must be a STRING or null."
                )

        headers = section.get("headers")
        if headers is not None and not isinstance(headers, dict):
            raise TypeError(
                f'OpenRouter connector "headers" for the {section_key} must be a '
                f"dict or null."
            )

        completion_kwargs = section.get("completion_kwargs") or {}
        if not isinstance(completion_kwargs, dict):
            raise TypeError(
                f'OpenRouter connector "completion_kwargs" for the {section_key} must '
                f"be a dict."
            )

        max_tokens = section.get("max_tokens")
        if max_tokens is None:
            max_tokens = 512

        return {
            "name": section["name"],
            "api_key": section["api_key"],
            "api_url": api_url,
            "site_url": section.get("site_url"),
            "site_name": section.get("site_name"),
            "headers": headers,
            "completion_kwargs": completion_kwargs,
            "max_tokens": max_tokens,
        }

    def generate(self, data: dict, multi_turn: bool = False) -> dict:
        """Generate a response from the target model via the OpenRouter API.

        Arguments:
            data: Dictionary containing data required for the generation API request.
                Valid Keys:
                    - prompt : str
                        Prompt for single turn generation. Required for single turn generation.
                    - messages: list[Message]
                        List of Message objects representing the conversation history.\
                        Message objects contain 'role' and 'content' attributes.\
                        Required for multi-turn conversation.
                    - system_prompt : str
                        Optional system prompt that defines the model's behavior, role, or constraints.
                    - temperature : float [0, 1]
                        Optional temperature setting for the target model. Defaults to 0.5 if not set.
                    - max_tokens : int
                        Optional setting for maximum generated tokens. Defaults to 512 if not set.
            multi_turn: Boolean flag to indicate if engaging in a multi turn conversation\
                with the target model. Default False.

        Returns:
            Generated response in format: {"response": str}

        Raises:
            KeyError: If a required key is missing from data.
            ValueError: If a value in data is of a wrong type.
            RuntimeError: If the API call fails.
        """
        if "temperature" not in data:
            data["temperature"] = 0.5
        data["max_tokens"] = self.max_tokens

        if "system_prompt" in data:
            if not isinstance(data["system_prompt"], str):
                raise ValueError(
                    'If using "system_prompt" in data, it needs to be a string.'
                )

        if multi_turn:
            if "messages" not in data:
                raise KeyError(
                    'Multi-turn conversation requires a "messages" key in \
                               data variable, which contains a List of Message objects \
                               representing the conversation history.'
                )
            if not isinstance(data["messages"], list):
                raise ValueError(
                    'Multi-turn conversation requires a "messages" key in \
                               data variable, which contains a List of Message objects \
                               representing the conversation history.'
                )
            for message in data["messages"]:
                if not isinstance(message, Message):
                    raise ValueError(
                        'Multi-turn conversation requires a "messages" key in \
                               data variable, which contains a List of Message objects \
                               representing the conversation history.'
                    )
            return self._multi_turn(data=data)
        else:
            if "prompt" not in data:
                raise KeyError(
                    'Single-turn conversation requires a "prompt" key in \
                               data variable, which contains a prompt as a string.'
                )
            if not isinstance(data["prompt"], str):
                raise ValueError(
                    'Single-turn conversation requires a "prompt" key in \
                               data variable, which contains a prompt as a string.'
                )
            return self._single_turn(data=data)

    def _single_turn(self, data: dict) -> dict:
        """Make a single-turn generation.

        Arguments:
            data: Dictionary with required data for API request.

        Returns:
            {"response": str}
        """
        if "system_prompt" in data:
            messages = [
                {"role": "system", "content": data["system_prompt"]},
                {"role": "user", "content": data["prompt"]},
            ]
        else:
            messages = [{"role": "user", "content": data["prompt"]}]

        return self._chat_completion(messages=messages, data=data)

    def _multi_turn(self, data: dict) -> dict:
        """Make a multi-turn generation.

        Arguments:
            data: Dictionary with required data for API request.

        Returns:
            {"response": str}
        """
        messages = [
            {"role": msg.role, "content": msg.content} for msg in data["messages"]
        ]
        if "system_prompt" in data:
            messages.insert(0, {"role": "system", "content": data["system_prompt"]})

        return self._chat_completion(messages=messages, data=data)

    def _chat_completion(self, messages: list, data: dict) -> dict:
        """Send a Chat Completions request to OpenRouter and unwrap the response.

        Arguments:
            messages: List of {"role", "content"} dicts for the request.
            data: Dictionary with "temperature" and "max_tokens" already set.

        Returns:
            {"response": str}

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=data["temperature"],
                max_tokens=data["max_tokens"],
                # OpenRouter-specific fields (e.g. "provider", "models", "transforms")
                # aren't part of the OpenAI SDK's typed kwargs, so they're passed
                # through via extra_body rather than as direct keyword arguments.
                extra_body=self.completion_kwargs,
            )
            choice = response.choices[0]
            return {"response": choice.message.content or ""}

        except Exception as e:
            logger.error(
                f"{ansi_colors['red']}ERROR while generating response from OpenRouter: {e}{ansi_colors['reset']}"
            )
            raise RuntimeError("Failed to generate response from OpenRouter.") from e

    def status_check(self) -> bool:
        """Check if the connector can reach the OpenRouter API and the target model is available.

        Returns:
            True if API is reachable and the target model exists.

        Raises:
            ConnectionError: If the API is not reachable.
        """
        try:
            model_ids = self._list_models()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to OpenRouter API at {self.base_url}: {e}")

        logger.info(f"Available models found: {len(model_ids)} models")

        if self.model in model_ids:
            logger.info(f"Model '{self.model}' is available.")
            return True

        # OpenRouter's catalog changes frequently and some model slugs (e.g.
        # provider-specific variants) may not appear in the listing but still work.
        logger.warning(
            f"Model '{self.model}' not found in available models list. "
            f"Proceeding anyway as some models may not be listed."
        )
        return True

    def _list_models(self) -> List[str]:
        """Helper method, used by status_check() to verify model availability.

        Returns:
            List of model ids.
        """
        models = self.client.models.list()
        return [m.id for m in models.data]
