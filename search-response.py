from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import chromadb
import torch
from llama_cpp import Llama

from utils import logger
from utils.constants import CHROMA_DB, CHROMA_COLLECTION, LOGGER
from utils.model_embeddings import get_embedding_model_chroma, get_model_name_llm
from utils.duration_decorator import measure_time

USE_GPU = True

has_cuda = torch.cuda.is_available()
print(f"Support GPU : {has_cuda}")
if has_cuda:
    print(f"Version CUDA : {torch.version.cuda}")

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
    _model = _model_init()
    _collection = _chroma_load()

    return _model, _collection

def _cuda_or_cpu():
    # return "cuda" if has_cuda and USE_GPU else "cpu"
    return "cpu"

@measure_time
def _model_init():
    """
    Selon CUDA avec PyTorch, initialisera avec GPU CUDA ou CPU
    """
    # import bitsandbytes as bnb
    # print(bnb.__version__)
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    # model = AutoModelForCausalLM.from_pretrained(get_model_name_llm(),
    #                                             # quantization_config=quantization_config,
    #                                             device_map="cpu")
    # print(model.hf_device_map)
    # return model
    return AutoModelForCausalLM.from_pretrained(get_model_name_llm(), torch_dtype=torch.float16).to(_cuda_or_cpu())
    # return model.to(_cuda_or_cpu())
@measure_time
def generate_response_nlp(prompt, model):
    """ Génération de la réponse Llama 2 """
    tokenizer = _get_tokenizer()
    logger.get_logger(LOGGER).debug(f"Tokenization de la tokenisation")
    inputs = tokenizer(prompt, return_tensors="pt").to(_cuda_or_cpu())
    logger.get_logger(LOGGER).debug(f"Génération réponse avec le modèle {model}")
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
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_texts = " ".join([doc for doc in results["documents"][0]])
    prompt = f"Extraits de ChromaDb trouvés : {retrieved_texts}\n\nQuestion : {question}\nRéponse :"

    return generate_response_nlp(prompt, model)

if __name__ == '__main__':
    question = "Quels sont les points clés du document ?"
    print(search_responses(question))
