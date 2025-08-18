import fitz  # PyMuPDF
import os
from pathlib import Path
import base64
from google import genai

client = genai.Client(api_key=)
OUTPUT_DIR = "parsed_pages"

def bboxes_overlap(a, b):
    """Check if two bounding boxes overlap at all."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1])


def image_to_caption(image_bytes, context_text=None, model="gemini-2.0-flash"):

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    user_prompt = "Please provide a concise description of this image."
    if context_text:
        user_prompt += f"\n\nContext text to guide you: {context_text}"

    response = client.models.generate_content(
        model=model,
        contents=[
            {"parts": [
                {"text": "You are an assistant that captions PDF figures and diagrams."},
                {"text": user_prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}},
            ]}
        ],
    )
    
    return response.text.strip()

# This File parses the input file and converts it into a set of MD files (one for each page) 
# This will contain text from that page as well as verbal descriptions of any images in it.
def parse_pdf_to_markdown(pdf_path):
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc, start=1): # type:ignore
        text_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        images = page.get_images(full=True)

        combined = []

        # Collect image bounding boxes
        image_boxes = {}
        for img in images:
            image_boxes[page.get_image_bbox(img)]=[]

        used_blocks = set()

        # For each image get all the text blocks that overlap with it.
        # Context text is used to help model understand the image better
        # Getting context text is also necessary to not include it in main text body.
        # This block can skipped if the book doesn't have OCR on the images. It's O(num_imgs * num_text_blocks). 
        for i, text_block in enumerate(text_blocks):
            bbox, text = text_block[:4], text_block[4].strip()
            for img_bbox in image_boxes.keys():    
                if text and bboxes_overlap(bbox, img_bbox):
                    image_boxes[img_bbox].append(text)
                    used_blocks.add(i)
                    break

        for img_bbox, overlapping_texts in image_boxes.items():
            # Crop image along with the text in it 
            pix = page.get_pixmap(clip=img_bbox)
            img_bytes = pix.tobytes("png")

            context_text = " ".join(overlapping_texts) if overlapping_texts else None
            caption = image_to_caption(img_bytes, context_text=context_text)

            combined.append((img_bbox, "image", caption))

        # Add text blocks not absorbed into images
        for i, text_block in enumerate(text_blocks):
            if i in used_blocks:
                continue
            bbox, text = text_block[:4], text_block[4].strip()
            if text:
                combined.append((bbox, "text", text))

        # Sort reading order: top-to-bottom, left-to-right
        combined.sort(key=lambda x: (x[0][1], x[0][0]))

        # Build Markdown
        md_content = []
        for _, kind, content in combined:
            if kind == "text":
                md_content.append(content)
            elif kind == "image":
                md_content.append(f"[Image: {content}]")

        md_text = "\n\n".join(md_content)

        # Save page markdown
        md_filename =  f"{OUTPUT_DIR}/page_{page_num:04d}.md"
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(md_text)

        print(f"Page {page_num} saved at {md_filename}")

    doc.close()

if __name__ == "__main__":
    input_pdf = "Introduction_to_probability.pdf"
    if not Path(input_pdf).exists():
        print(f"Input PDF file '{input_pdf}' does not exist.")
    else:
        parse_pdf_to_markdown(input_pdf)
        print("PDF parsing complete.")