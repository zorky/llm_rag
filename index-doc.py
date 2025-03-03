import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.constants import CHROMA_DB, CHROMA_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import init_logger
from utils.model_embeddings import get_embedding_model_chroma
from utils.measure_time_decorator import measure_time

KB_DOC = "kb/harcelement-ecole.pdf"
IDX_PREFIX = "doc_"

@measure_time
def create_db():
    """
    Init ChromaDB et indexe le texte d'un document PDF avec embeddings (vecteurs)
    """
    embedding_model = get_embedding_model_chroma()

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB)
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

    # Charger et découper un document PDF
    pdf_loader = PyPDFLoader(KB_DOC)
    pages = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(pages)

    # Indexation des passages dans ChromaDB
    for i, chunk in enumerate(texts):
        embedding_vector = embedding_model.embed_query(chunk.page_content)
        collection.add(
            ids=[f"{IDX_PREFIX}{i}"],
            documents=[chunk.page_content],
            embeddings=[embedding_vector],
            metadatas=[{"source": KB_DOC}]
        )

    print(f"Indexation terminée : {len(texts)} chunks ajoutés.")

if __name__ == '__main__':
    init_logger()
    create_db()