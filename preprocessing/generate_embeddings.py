import os
from langchain.docstore.document import Document
from semantic_text_splitter import MarkdownSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from dotenv import load_dotenv
import time
from langchain_google_genai._common import GoogleGenerativeAIError
from preprocessing.simple_rate_limiter import SimpleRateLimiter


def generate_embeddings(file_name):
    load_dotenv()
    BASE_DIR = Path(__file__).resolve().parent
    out_path = BASE_DIR.parent / "outputs" / file_name.split(".")[0]
    chunk_size = 500
    MAX_RETRIES = 6
    BATCH_SIZE = 30
    rate_limiter = SimpleRateLimiter(
        requests_per_minute=4 * 30000 / (chunk_size * BATCH_SIZE)
    )
    try:
        sections = os.listdir(out_path)
    except FileNotFoundError:
        print(f"[ERROR] '{file_name}' has not been parsed yet.")
        return

    if os.path.exists(out_path / "index"):
        print(f"[ERROR] '{file_name}' has already been embedded.")
        return

    all_chunks = []

    for section in sections:
        with open(out_path / section, "r", encoding="utf-8") as f:
            content = f.read()

        splitter = MarkdownSplitter(chunk_size)
        text_chunks = splitter.chunks(content)

        if not text_chunks:
            print(f"Warning: No chunks created for '{section}'. Skipping.")
            continue

        for chunk in text_chunks:
            all_chunks.append(
                Document(
                    page_content=chunk, metadata={"source": str(out_path / section)}
                )
            )

        print(f"Processed '{section}' and found {len(text_chunks)} chunks.")

    if not all_chunks:
        print("No chunks were processed. No embeddings will be created.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", task_type="QUESTION_ANSWERING"
    )
    first_batch = all_chunks[:BATCH_SIZE]
    remaining_chunks = all_chunks[BATCH_SIZE:]

    print(f"Creating initial FAISS index with {len(first_batch)} chunks.")
    db = FAISS.from_documents(documents=first_batch, embedding=embeddings)

    if remaining_chunks:
        print(
            f"Adding remaining {len(remaining_chunks)} chunks in batches of {BATCH_SIZE}."
        )
        for i in range(0, len(remaining_chunks), BATCH_SIZE):
            batch = remaining_chunks[i : i + BATCH_SIZE]
            retries = 0
            current_delay = 1
            while True:
                try:
                    db.add_documents(batch)
                    print(f"Successfully added batch {i//BATCH_SIZE + 2}.")
                    rate_limiter.wait_if_needed()
                    break
                except GoogleGenerativeAIError as e:
                    print(
                        f"[ERROR] Rate limit likely hit. (Actual Error: {e}) Retrying in {current_delay} seconds..."
                    )
                    if retries >= MAX_RETRIES:
                        break
                    time.sleep(current_delay)
                    retries += 1
                    current_delay *= 2  # Exponential backoff

                except Exception as e:
                    print(f"[ERROR] An unexpected error occurred: {e}")
                    break  # Break for other errors

    db.save_local(out_path / "index")
    print("FAISS index saved successfully.")
