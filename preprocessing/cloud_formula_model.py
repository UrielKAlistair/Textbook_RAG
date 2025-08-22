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
from google.genai.errors import ClientError, ServerError
from dotenv import load_dotenv
import os
from io import BytesIO
from preprocessing.simple_rate_limiter import SimpleRateLimiter
import time


class CloudFormulaModel(CodeFormulaModel):
    def __init__(self, enabled: bool):
        """
        A cloud-backed replacement for CodeFormulaModel.
        """
        self.enabled = enabled
        self.options = CodeFormulaModelOptions()
        self.rate_limiter = SimpleRateLimiter(requests_per_minute=10)

        load_dotenv()
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key).models

        self.models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
        ]  # Tried in Order
        self.max_retries = 6

        self.code_prompt = """Return only the exact source code. 
        Do not include explanations, text, or markdown fences. 
        Preserve indentation and syntax. 
        Detect and keep the correct programming language.
        """
        self.formula_prompt = """Return only the mathematical expression in valid LaTeX format. 
        Do not include explanations, text, or markdown fences. 
        Output must be raw LaTeX suitable for compiling.
        """

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            yield from (el.item for el in element_batch)
            return

        for el in element_batch:
            item = el.item
            if not isinstance(item, (CodeItem, FormulaItem)):
                yield item
                continue

            if isinstance(item, CodeItem):
                output = self.get_gemini_description(el.image, code=True)
                output, code_language = self._extract_code_language(output)
                item.code_language = self._get_code_language_enum(code_language)
            else:
                output = self.get_gemini_description(el.image, code=False)
            item.text = output
            yield item

    def get_gemini_description(self, image, code=True):
        self.rate_limiter.wait_if_needed()
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        query = self.code_prompt if code else self.formula_prompt

        if not self.models:
            return ""

        for index, model_name in enumerate(self.models):
            for attempt in range(1, self.max_retries + 1):
                try:
                    print(
                        f"[INFO] Attempting model {model_name} (attempt {attempt}/{self.max_retries + 1})..."
                    )
                    resp = self.client.generate_content(
                        contents=[
                            types.Part.from_bytes(
                                data=image_bytes, mime_type="image/png"
                            ),
                            query,
                        ],
                        model=model_name,
                    )

                    if index != 0:
                        self.models.insert(0, self.models.pop(index))
                    return resp.text.strip() if resp and resp.text else ""

                except ServerError or ClientError as e:
                    # This handles all transient, retriable errors.
                    if attempt <= self.max_retries:
                        print(
                            f"[WARN] Call to {model_name} failed (attempt {attempt}/{self.max_retries}): {e}. Retrying in {2**attempt} seconds..."
                        )
                        time.sleep(2**attempt)
                    else:
                        print(
                            f"[ERROR] All retries failed for {model_name}. Removing {model_name} and trying others."
                        )
                        break

                except Exception as e:
                    print(
                        f"[ERROR] Unexpected error with model {model_name}: {e}. Skipping to next model..."
                    )
                    break

        print("[FATAL] All model calls failed. Returning empty string.")
        return ""
