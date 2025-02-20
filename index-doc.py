import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import logging
logging.basicConfig(level=logging.DEBUG)

# Configuration du modèle d'embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialisation de ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="docs")

# Charger et découper un document PDF
pdf_loader = PyPDFLoader("kb/document.pdf")
pages = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(pages)

# Indexation des passages dans ChromaDB
for i, chunk in enumerate(texts):
    collection.add(
        ids=[f"doc_{i}"],
        documents=[chunk.page_content],
        metadatas=[{"source": "document.pdf"}]
    )

print(f"Indexation terminée : {len(texts)} chunks ajoutés.")