import chromadb
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders.pdf import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.constants import CHROMA_DB, CHROMA_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import init_logger
from utils.model_embeddings import get_embedding_model_chroma
from utils.measure_time_decorator import measure_time

KB_DIR = "kb"
KB_DOC_EXAMPLE = f"{KB_DIR}/harcelement-ecole.pdf"
IDX_PREFIX = "doc_"
LOAD_ALL_DOCS = True

def load_documents():
    document_loader = PyPDFDirectoryLoader(KB_DIR)
    return document_loader.load()

def _get_documents_or_pages(load_all_docs) -> list[Document]:
    if load_all_docs:
        documents = load_documents()
    else:
        pdf_loader = PyPDFLoader(KB_DOC_EXAMPLE)
        documents = pdf_loader.load()
    return documents

@measure_time
def create_db(load_all_docs=True or LOAD_ALL_DOCS):
    """
    Init ChromaDB et indexe le texte d'un document PDF avec embeddings (vecteurs)
    """
    embedding_model = get_embedding_model_chroma()

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB)
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

    documents = _get_documents_or_pages(load_all_docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    # Indexation des passages dans ChromaDB
    for i, chunk in enumerate(texts):
        if i < 50: # limit car sur ma station ça plante segmentation fault
            embedding_vector = embedding_model.embed_query(chunk.page_content)
            print(f"{embedding_vector}")
            collection.add(
                ids=[f"{IDX_PREFIX}{i}"],
                documents=[chunk.page_content],
                embeddings=[embedding_vector],
                metadatas=[{"source": KB_DOC_EXAMPLE}]
            )
        else:
            break

    print(f"Indexation terminée : {len(texts)} chunks ajoutés.")

if __name__ == '__main__':
    init_logger()
    create_db(load_all_docs=False)