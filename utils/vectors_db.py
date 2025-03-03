import chromadb
from chromadb.api.models import Collection

from utils.measure_time_decorator import measure_time
from utils.constants import CHROMA_DB, CHROMA_COLLECTION

@measure_time
def chroma_load() -> Collection:
    """
    Charge les collections déjà créés de vecteurs / embeddings de ChromaDB
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB)
    return chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)