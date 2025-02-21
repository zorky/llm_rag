from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
import torch

from utils import logger
from utils.model_embeddings import get_embedding_model
from utils.duration_decorator import measure_time

LOGGER = "raggy_logger"
CHROMA_DB = "./chroma_db2"

has_cuda = torch.cuda.is_available()
print(f"Support GPU : {has_cuda}")  # Doit renvoyer True si CUDA est bien installé
if has_cuda:
    print(f"Version CUDA : {torch.version.cuda}")  # Doit afficher la version de CUDA utilisée

USE_GPU = True

# Charger Llama 2
model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

@measure_time
def chroma_load():
    """
    Charge les collections de vecteurs / embeddings de ChromaDB
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB)
    collection = chroma_client.get_or_create_collection(name="docs")
    return collection

@measure_time
def model_init():
    """
    Selon CUDA avec PyTorch, initialisera avec GPU CUDA ou CPU
    """
    if has_cuda and USE_GPU:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
    else:
        # force le CPU vs GPU CUDA
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cpu")
    return model

@measure_time
def generate_response(question, model, collection):
    """
    Recherche des vecteurs dans ChromaDB et génération de réponse avec Llama 2
    """
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(question)
    results = collection.query(
        # query_texts=[question], # recherche textuelle simple (mode BM25)
        query_embeddings=[query_embedding], # optionnel si ChromaDb a été initialisé avec des embeddings pour une recherche sémantique
        n_results=3
    )

    retrieved_texts = " ".join([doc for doc in results["documents"][0]])
    prompt = f"Voici des informations utiles : {retrieved_texts}\n\nQuestion : {question}\nRéponse :"

    # Génération tokens avec Llama 2
    if has_cuda and USE_GPU:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    else:
        # force le CPU
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    # ValueError: Input length of input_ids is 460, but `max_length` is set to 200. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.
    # output = model.generate(**inputs, max_length=200)
    output = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

def _init_logger():
    logger.setup_logging()
    log = logger.get_logger(LOGGER)
    return log

def init_llm():
    _model = model_init()
    _collection = chroma_load()

    return _model, _collection

if __name__ == '__main__':
    _init_logger()
    _model, _collection = init_llm()
    question = "Quels sont les points clés du document ?"
    print(generate_response(question, _model, _collection))
    # question = input("Poser une question : ")
    # print(generate_response(question))