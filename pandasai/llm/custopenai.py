"""OpenAI LLM API

This module is to run the OpenAI API using OpenAI API.

Example:
    Use below example to call OpenAI Model

    >>> from pandasai.llm.openai import OpenAI
"""

import os
from typing import Any, Dict, Optional

import openai
from ..helpers import load_dotenv

from ..exceptions import APIKeyNotFoundError, UnsupportedModelError
from ..helpers.openai import is_openai_v1
from .base import BaseOpenAI

load_dotenv()


class CustOpenAI(BaseOpenAI):
    """OpenAI LLM using BaseOpenAI Class.

    An API call to OpenAI API is sent and response is recorded and returned.
    The default chat model is **gpt-3.5-turbo**.
    The list of supported Chat models includes ["gpt-4", "gpt-4-0613", "gpt-4-32k",
     "gpt-4-32k-0613", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613",
     "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-instruct"].
    The list of supported Completion models includes "gpt-3.5-turbo-instruct" and
     "text-davinci-003" (soon to be deprecated).
    """

    _supported_chat_models = [
	"local-model"
    ]
    _supported_completion_models = ["local-completion-model"]

    model: str = "local-model"

    def __init__(
        self,
        api_base: str = "http://172.16.180.28:1234/v1", 
        api_token: str = "OPENAI_API_TOKEN",
        stop: str = "### Instruction:",
        max_tokens: int = 2048
        ** kwargs,
    ):
        """
        __init__ method of OpenAI Class

        Args:
            api_token (str): API Token for OpenAI platform. Here is an example: "http://172.16.180.28:1234/v1"
            **kwargs: Extended Parameters inferred from BaseOpenAI class (and LLM class)
            stop: by default it is "### Instruction:" which is used to stop the chat (Llama format)

        """
        self.api_base = api_base or None
        self.api_token = self.api_token
        self.stop = stop or "### Instruction:"
        self.max_tokens = max_tokens or 1024
        self.openai_proxy = kwargs.get("openai_proxy") or os.getenv("OPENAI_PROXY")

        if self.openai_proxy:
            openai.proxy = {"http": self.openai_proxy, "https": self.openai_proxy}

        self._set_params(**kwargs)

        # set the client
        model_name = self.model.split(":")[1] if "ft:" in self.model else self.model
        if model_name in self._supported_chat_models:
            self._is_chat_model = True
            openai.api_key = ""
            openai.api_base = self.api_base
            self.client = (
                openai.OpenAI(**self._client_params).chat.completions
                if is_openai_v1()
                else openai.ChatCompletion
            )
        elif model_name in self._supported_completion_models:
            self._is_chat_model = False
            openai.api_key = ""
            openai.api_base = self.api_base
            self.client = (
                openai.OpenAI(**self._client_params).completions
                if is_openai_v1()
                else openai.Completion
            )
        else:
            raise UnsupportedModelError(self.model)

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API"""
        return {
            **super()._default_params,
            "model": self.model,
        }

    @property
    def type(self) -> str:
        return "openai"
