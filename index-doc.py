import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.constants import CHROMA_DB, CHROMA_COLLECTION, LOGGER
from utils.duration_decorator import measure_time
from utils import logger

KB_DOC = "kb/harcelement-ecole.pdf"
IDX_PREFIX = "doc_"

@measure_time
def create_db():
    """
    Init ChromaDB et indexe le texte d'un document PDF avec embeddings (vecteurs)
    """
    _init_logger()

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB)
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

    # Charger et découper un document PDF
    pdf_loader = PyPDFLoader(KB_DOC)
    pages = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(pages)

    # Indexation des passages dans ChromaDB
    for i, chunk in enumerate(texts):
        collection.add(
            ids=[f"{IDX_PREFIX}{i}"],
            documents=[chunk.page_content],
            metadatas=[{"source": KB_DOC}]
        )

    print(f"Indexation terminée : {len(texts)} chunks ajoutés.")

def _init_logger():
     logger.setup_logging()
     log = logger.get_logger(LOGGER)
     return log

if __name__ == '__main__':
    create_db()