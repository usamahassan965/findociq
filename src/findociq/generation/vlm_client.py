"""Stage 4 - Generation: answer over retrieved page images with a VLM.

One interface, three providers:
  gemini    - Google AI Studio free tier (default; get a key at aistudio.google.com)
  ollama    - fully local and free (qwen2.5vl), needs ~8 GB VRAM for the 7b
  anthropic - paid, highest answer quality (Claude)

The VLM receives the actual page images, so it reads charts and tables
directly instead of a lossy OCR transcript.
"""

import base64
from abc import ABC, abstractmethod
from pathlib import Path

import requests

from findociq.config import get_settings
from findociq.retrieval.hybrid_retriever import RetrievedPage

SYSTEM_PROMPT = (
    "You are a financial document analyst. Answer the user's question using ONLY the "
    "provided document page images. Read charts, tables, and figures directly from the "
    "images. Cite every claim with the source in the form [doc_name p.N]. If the pages "
    "do not contain the answer, say so explicitly instead of guessing."
)


def _build_user_text(question: str, pages: list[RetrievedPage]) -> str:
    page_list = "\n".join(f"- Image {i + 1}: {p.doc_name} p.{p.page_number}" for i, p in enumerate(pages))
    return f"Document pages provided:\n{page_list}\n\nQuestion: {question}"


class VLMClient(ABC):
    @abstractmethod
    def answer(self, question: str, pages: list[RetrievedPage]) -> str: ...


class GeminiClient(VLMClient):
    def __init__(self):
        from google import genai

        settings = get_settings()
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set - get a free key at aistudio.google.com")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    def answer(self, question: str, pages: list[RetrievedPage]) -> str:
        from google.genai import types

        parts = [
            types.Part.from_bytes(data=Path(p.image_path).read_bytes(), mime_type="image/png")
            for p in pages
        ]
        parts.append(types.Part.from_text(text=_build_user_text(question, pages)))
        response = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return response.text or ""


class OllamaClient(VLMClient):
    def __init__(self):
        settings = get_settings()
        self.url = settings.ollama_url
        self.model = settings.ollama_model

    def answer(self, question: str, pages: list[RetrievedPage]) -> str:
        images = [
            base64.b64encode(Path(p.image_path).read_bytes()).decode() for p in pages
        ]
        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _build_user_text(question, pages),
                        "images": images,
                    },
                ],
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class AnthropicClient(VLMClient):
    def __init__(self):
        import anthropic

        settings = get_settings()
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model

    def answer(self, question: str, pages: list[RetrievedPage]) -> str:
        content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(Path(p.image_path).read_bytes()).decode(),
                },
            }
            for p in pages
        ]
        content.append({"type": "text", "text": _build_user_text(question, pages)})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        return next((block.text for block in response.content if block.type == "text"), "")


def get_vlm_client() -> VLMClient:
    provider = get_settings().vlm_provider.lower()
    if provider == "gemini":
        return GeminiClient()
    if provider == "ollama":
        return OllamaClient()
    if provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"Unknown VLM_PROVIDER: {provider}")
