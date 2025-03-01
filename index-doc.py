import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma

from utils.constants import CHROMA_DB, CHROMA_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP, LOGGER
from utils.duration_decorator import measure_time
from utils import logger
from utils.model_embeddings import get_embeddings_model

KB_DOC = "kb/harcelement-ecole.pdf"
IDX_PREFIX = "doc_"

def _chroma_embedding():
    """
    Charge le modèle de vecteurs de Chroma
    """
    embedding_model = get_embeddings_model()
    chroma = Chroma(persist_directory=CHROMA_DB, embedding_function=embedding_model)
    return chroma

def _chunks_doc():
    """
    Charge et découpe un document PDF
    """
    pdf_loader = PyPDFLoader(KB_DOC)
    pages = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP,
                                                   length_function=len,
                                                   add_start_index=True
    )
    texts = text_splitter.split_documents(pages)
    return texts

@measure_time
def save_doc_to_chroma():
    """
    Sauvegarde le modèle de vecteurs de Chroma
    """
    chroma = _chroma_embedding()
    chunks = _chunks_doc()
    chroma.add_documents(chunks)
    print(f"Indexation terminée : {len(chunks)} chunks ajoutés.")

def _init_logger():
     logger.setup_logging()
     log = logger.get_logger(LOGGER)
     return log

if __name__ == '__main__':
    _init_logger()
    save_doc_to_chroma()