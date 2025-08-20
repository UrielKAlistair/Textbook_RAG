from pathlib import Path
import time, os
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from preprocessing.cloud_formula_model import CloudFormulaModel
from docling_core.types.doc.document import DoclingDocument, SectionHeaderItem, NodeItem
import re


def gemini_vlm_options():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    options = PictureDescriptionApiOptions(
        url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",  # type: ignore
        params={"model": "gemini-2.5-flash"},
        headers=headers,
        prompt=(
            "Describe the image with scientific precision in up to 5 concise sentences. "
            "Mention main objects, relationships, and any readable text."
        ),
        timeout=60,
    )
    return options


def parse_pdf(file_name):
    BASE_DIR = Path(__file__).resolve().parent
    pdf_path = BASE_DIR.parent / "inputs" / file_name
    out_path = BASE_DIR.parent / "outputs" / file_name.split(".")[0]

    opts = PdfPipelineOptions(enable_remote_services=True)
    opts.images_scale = 1
    opts.do_ocr = False
    opts.do_picture_description = True
    opts.do_formula_enrichment = True
    opts.picture_description_options = gemini_vlm_options()

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    converter._get_pipeline(InputFormat.PDF).enrichment_pipe[0] = CloudFormulaModel(enabled=opts.do_formula_enrichment)  # type: ignore

    print(f"Starting conversion")
    t0 = time.time()
    result = converter.convert(pdf_path)
    print(f"Conversion finished in {time.time() - t0:.1f}s")
    save_pdf(result.document, out_path)


def save_pdf(result: DoclingDocument, out_path: Path):
    out_path.mkdir(parents=True, exist_ok=True)
    md_text = result.export_to_markdown()

    # regex to find headers and their content
    pattern = re.compile(r"^(#+ .+)$", re.MULTILINE)

    sections = []
    last_pos = 0
    last_header = None

    for match in pattern.finditer(md_text):
        if last_header is not None:
            # content between previous header and this header
            content = md_text[last_pos : match.start()].strip()
            sections.append((last_header, content))
        last_header = match.group(1)
        last_pos = match.end()

    # add final section
    if last_header is not None:
        content = md_text[last_pos:].strip()
        sections.append((last_header, content))

    # merge empty sections into next non-empty
    merged_sections = []
    buffer_header = None
    for header, content in sections:
        if not content:  # empty section
            if buffer_header is None:
                buffer_header = header
            else:
                buffer_header += " + " + header  # keep track of merged headers
        else:
            if buffer_header:
                # merge the buffered empty header(s) into this one
                header = buffer_header + " + " + header
                buffer_header = None
            merged_sections.append((header, content))

    # write each section to its own file
    for i, (header, content) in enumerate(merged_sections, start=1):
        fname = f"section_{i}"
        file_path = out_path / f"{fname}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(header + "\n\n" + content)
            print(f"Saved section {fname} to {file_path.resolve()}")
