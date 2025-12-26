"""
Ollama Client for Local LLM Integration
=======================================

This module provides a client for interacting with Ollama local language models.
It replaces OpenAI API calls with local Ollama API calls for complete independence
from external services.

Key Features:
- Local LLM integration via Ollama
- JSON response formatting
- Error handling and retries
- Model management
- OpenAI-compatible wrapper for easy migration
"""

import json
import logging
import requests
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Response from Ollama API."""
    content: str
    model: str
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None


class OllamaClient:
    """
    Client for interacting with Ollama local language models.

    This client provides a similar interface to OpenAI's client but uses
    local Ollama models instead of external API calls.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.1:latest",
        timeout: int = 120,
        auto_test_connection: bool = True
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            default_model: Default model to use
            timeout: Request timeout in seconds
            auto_test_connection: Whether to test connection on init
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.timeout = timeout

        if auto_test_connection:
            self._test_connection()

    def _test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                logger.info(f"Connected to Ollama. Available models: {available_models}")

                # Check if default model is available
                if self.default_model not in available_models:
                    logger.warning(
                        f"Default model '{self.default_model}' not found. "
                        f"Available: {available_models}"
                    )
                    if available_models:
                        self.default_model = available_models[0]
                        logger.info(f"Using '{self.default_model}' as default model")
                return True
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama server: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            return False

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None
    ) -> OllamaResponse:
        """
        Generate response using Ollama.

        Args:
            prompt: User prompt
            model: Model name (uses default if None)
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            format: Response format ('json' for JSON)

        Returns:
            OllamaResponse object
        """
        model = model or self.default_model

        # Prepare request data
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        # Add system prompt if provided
        if system:
            data["system"] = system

        # Add max tokens if provided
        if max_tokens:
            data["options"]["num_predict"] = max_tokens

        # Add format if provided
        if format == "json":
            data["format"] = "json"

        try:
            logger.debug(f"Sending request to Ollama: {model}")
            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                duration = time.time() - start_time
                logger.debug(f"Ollama response received in {duration:.2f}s")

                return OllamaResponse(
                    content=result.get('response', ''),
                    model=model,
                    total_duration=result.get('total_duration'),
                    load_duration=result.get('load_duration'),
                    prompt_eval_count=result.get('prompt_eval_count'),
                    eval_count=result.get('eval_count')
                )
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.error("Ollama request timeout")
            raise Exception("Ollama request timeout")
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            raise

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> OllamaResponse:
        """
        Chat completion compatible with OpenAI format.

        Args:
            messages: List of message dictionaries
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            response_format: Response format specification

        Returns:
            OllamaResponse object
        """
        # Convert messages to prompt format
        prompt_parts = []
        system_prompt = None

        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')

            if role == 'system':
                system_prompt = content
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")

        # Combine into single prompt
        prompt = "\n".join(prompt_parts)
        if not prompt.endswith("Assistant:"):
            prompt += "\nAssistant:"

        # Determine format
        format_type = None
        if response_format and response_format.get('type') == 'json_object':
            format_type = "json"

        return self.generate(
            prompt=prompt,
            model=model,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            format=format_type
        )

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            else:
                logger.error(f"Failed to list models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def pull_model(self, model_name: str, timeout: int = 600) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model_name: Name of the model to pull
            timeout: Timeout in seconds (default 10 minutes)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model: {model_name}")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=timeout
            )

            if response.status_code == 200:
                logger.info(f"Model {model_name} pulled successfully")
                return True
            else:
                logger.error(f"Failed to pull model: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False


# ==================== OpenAI-Compatible Wrappers ====================

class OpenAICompatibleClient:
    """
    OpenAI-compatible wrapper for Ollama client.

    This class provides the same interface as OpenAI's client
    but uses Ollama underneath for seamless migration.
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        """Initialize with Ollama client."""
        self.ollama_client = ollama_client or OllamaClient()
        self.chat = ChatCompletions(self.ollama_client)


class ChatCompletions:
    """Chat completions endpoint wrapper."""

    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client

    def create(self, **kwargs) -> 'MockOpenAIResponse':
        """Create chat completion."""
        response = self.ollama_client.chat_completion(**kwargs)
        return MockOpenAIResponse(response)


class MockOpenAIResponse:
    """Mock OpenAI response format."""

    def __init__(self, ollama_response: OllamaResponse):
        self.choices = [MockChoice(ollama_response)]


class MockChoice:
    """Mock OpenAI choice format."""

    def __init__(self, ollama_response: OllamaResponse):
        self.message = MockMessage(ollama_response.content)


class MockMessage:
    """Mock OpenAI message format."""

    def __init__(self, content: str):
        self.content = content


# ==================== Utility Functions ====================

def setup_recommended_models() -> List[str]:
    """Setup recommended models for academic RAG."""
    client = OllamaClient(auto_test_connection=False)

    recommended_models = [
        "llama3.1:latest",
        "gpt-oss:latest",
        "llama3.1:8b"
    ]

    available_models = client.list_models()

    for model in recommended_models:
        if model not in available_models:
            logger.info(f"Recommended model {model} not found. Consider pulling it:")
            logger.info(f"ollama pull {model}")

    return available_models


def test_ollama_setup() -> bool:
    """Test Ollama setup and performance."""
    print("🧪 Testing Ollama Setup")
    print("=" * 30)

    try:
        client = OllamaClient()

        # Test basic generation
        response = client.generate(
            prompt="What is 2+2? Answer briefly.",
            temperature=0.1
        )

        print(f"Basic generation test:")
        print(f"   Model: {response.model}")
        print(f"   Response: {response.content.strip()}")

        # Test JSON format
        json_response = client.generate(
            prompt="Return a JSON object with 'answer': '4' for the question: What is 2+2?",
            format="json",
            temperature=0.1
        )

        print(f"JSON format test:")
        print(f"   Response: {json_response.content.strip()}")

        # Test chat completion
        chat_response = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            temperature=0.1
        )

        print(f"Chat completion test:")
        print(f"   Response: {chat_response.content.strip()}")

        print(f"\nAvailable models: {client.list_models()}")
        return True

    except Exception as e:
        print(f"Ollama test failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        return False


if __name__ == "__main__":
    test_ollama_setup()
