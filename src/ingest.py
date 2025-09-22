import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

for k in ("GOOGLE_API_KEY", "DATABASE_URL","PG_VECTOR_COLLECTION_NAME","PDF_PATH"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

PDF_PATH = os.getenv("PDF_PATH")


def ingest_pdf():
  pp = Path(PDF_PATH) / "document.pdf"
  docs = PyPDFLoader(str(pp)).load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  chunks = text_splitter.split_documents(docs)
  enriched = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    )
    for d in chunks
  ]
  ids = [f"doc-{i}" for i in range(len(enriched))]
  
  embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL"), 
    api_key=os.getenv("GOOGLE_API_KEY")
  )
  
  pgvector = PGVector(
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    embeddings=embeddings,
    connection=os.getenv("DATABASE_URL"),
    use_jsonb=True,
  )
  pgvector.add_documents(documents=enriched, ids=ids)


if __name__ == "__main__":
    ingest_pdf()