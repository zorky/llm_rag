from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
import torch

from utils import logger
from utils.constants import CHROMA_DB, CHROMA_COLLECTION, LOGGER
from utils.model_embeddings import get_embedding_model_chroma, get_model_name_llm
from utils.duration_decorator import measure_time

USE_GPU = True

has_cuda = torch.cuda.is_available()
print(f"Support GPU : {has_cuda}")  # Doit renvoyer True si CUDA est bien installé
if has_cuda:
    print(f"Version CUDA : {torch.version.cuda}")  # Doit afficher la version de CUDA utilisée

@measure_time
def _chroma_load():
    """
    Charge les collections de vecteurs / embeddings de ChromaDB
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB)
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
    return collection

def _get_tokenizer():
    return AutoTokenizer.from_pretrained(get_model_name_llm())

def _init_logger():
    logger.setup_logging()
    log = logger.get_logger(LOGGER)
    return log

def _init_llm():
    _model = model_init()
    _collection = _chroma_load()

    return _model, _collection

@measure_time
def model_init():
    """
    Selon CUDA avec PyTorch, initialisera avec GPU CUDA ou CPU
    """
    if has_cuda and USE_GPU:
        model = AutoModelForCausalLM.from_pretrained(get_model_name_llm(), torch_dtype=torch.float16).to("cuda")
    else:
        # force le CPU vs GPU CUDA
        model = AutoModelForCausalLM.from_pretrained(get_model_name_llm(), torch_dtype=torch.float16).to("cpu")
    return model

@measure_time
def generate_response_nlp(prompt, model):
    """ Génération de la réponse Llama 2 """

    tokenizer = _get_tokenizer()
    if has_cuda and USE_GPU:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    # ValueError: Input length of input_ids is 460, but `max_length` is set to 200. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.
    # output = model.generate(**inputs, max_length=200)
    output = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@measure_time
def search_responses(question):
    """
    Recherche des vecteurs dans ChromaDB et génération de réponse avec Llama 2
    """
    _init_logger()
    model, collection = _init_llm()

    embedding_model = get_embedding_model_chroma()
    query_embedding = embedding_model.embed_query(question)
    results = collection.query(
        # query_texts=[question], # recherche textuelle simple (mode BM25)
        query_embeddings=[query_embedding], # optionnel si ChromaDb a été initialisé avec des embeddings pour une recherche sémantique
        n_results=3
    )

    retrieved_texts = " ".join([doc for doc in results["documents"][0]])
    prompt = f"Extraits de ChromaDb trouvés : {retrieved_texts}\n\nQuestion : {question}\nRéponse :"

    return generate_response_nlp(prompt, model)

if __name__ == '__main__':
    question = "Quels sont les points clés du document ?"
    print(search_responses(question))
    # question = input("Poser une question : ")
    # print(search_responses(question))