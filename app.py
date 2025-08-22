from fastapi import FastAPI, Query
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from fastapi.responses import RedirectResponse
import urllib.parse

preprocess = False
file_name = "Introduction_to_probability-223-266.pdf"
index_path = f"outputs/{file_name.split(".")[0]}/index"
app = FastAPI()
load_dotenv()
PROMPT_TEMPLATE = """You are an assistant that answers questions using the provided documents. 
If the answer is not contained within the documents, say you are unable to assist with the user's query.

Question:
{question}

Relevant documents:
{context}

Answer:"""


if preprocess:
    from preprocessing.parse_input import parse_pdf
    from preprocessing.generate_embeddings import generate_embeddings

    parse_pdf(file_name)
    generate_embeddings(file_name)


def load_faiss_index(index_path: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", task_type="RETRIEVAL_QUERY"
    )
    faiss_index = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    return faiss_index


vectorstore = load_faiss_index(index_path)


@app.get("/query")
def query_docs(q: str = Query(..., description="User query string")):
    docs = vectorstore.similarity_search(q, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = PROMPT_TEMPLATE.format(question=q, context=context)
    encoded = urllib.parse.quote(prompt)
    return RedirectResponse(url=f"https://chat.openai.com/?q={encoded}")
