import chromadb

from utils import logger
from utils.constants import CHROMA_DB, CHROMA_COLLECTION, LOGGER
from utils.model_embeddings import get_embedding_model_chroma
from utils.duration_decorator import measure_time

@measure_time
def _chroma_load():
    """
    Charge les collections de vecteurs / embeddings de ChromaDB
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB)
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
    return collection

def _init_logger():
    logger.setup_logging()
    log = logger.get_logger(LOGGER)
    return log

@measure_time
def search_documents(question):
    """
    Recherche des vecteurs dans ChromaDB afin d'obtenir les documents approchants
    """
    _init_logger()
    collection = _chroma_load()

    embedding_model = get_embedding_model_chroma()
    query_embedding = embedding_model.embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_texts = " ".join([doc for doc in results["documents"][0]])
    print(f"Extraits de ChromaDb trouvés\n")
    print(f"Question : {question}\n\nDocuments : {retrieved_texts}\n\n")

if __name__ == '__main__':
    question = "Quels sont les points clés du document ?"
    print(search_documents(question))
