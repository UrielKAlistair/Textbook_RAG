from typing import Iterable
from docling_core.types.doc.document import (
    NodeItem,
    DoclingDocument,
    CodeItem,
    FormulaItem,
)
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.code_formula_model import CodeFormulaModel, CodeFormulaModelOptions
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from io import BytesIO
from preprocessing.simple_rate_limiter import SimpleRateLimiter

MODEL_NAME = "gemini-2.5-flash"


class CloudFormulaModel(CodeFormulaModel):
    def __init__(self, enabled: bool):
        """
        A cloud-backed replacement for CodeFormulaModel.
        """
        self.enabled = enabled
        self.options = CodeFormulaModelOptions()
        self.rate_limiter = SimpleRateLimiter(requests_per_minute=5)

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        for el in element_batch:
            item = el.item
            if not isinstance(item, (CodeItem, FormulaItem)):
                yield item
                continue
            print("Calling Gemini for formula/code extraction...")
            output = self.get_gemini_description(el.image)
            if isinstance(item, CodeItem):
                output, code_language = self._extract_code_language(output)
                item.code_language = self._get_code_language_enum(code_language)
            item.text = output
            yield item

    def get_gemini_description(self, image):
        self.rate_limiter.wait_if_needed()
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        model = MODEL_NAME

        query = f"Extract the formulae as LaTeX without commentary."

        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
                query,
            ],
        )
        return resp.text.strip() if resp and resp.text else ""
