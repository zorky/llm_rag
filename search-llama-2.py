from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import chromadb
import torch
import time

from utils import logger
from utils.constants import CHROMA_DB, CHROMA_COLLECTION, LOGGER
from utils.model_embeddings import get_embedding_model_chroma, get_model_name_llm
from utils.duration_decorator import measure_time

USE_GPU = True
USE_QUANTIZATION = False

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
    return "cuda" if has_cuda and USE_GPU else "cpu"

def _getConfigQuantification():
  quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # quantification en 4 bits
    bnb_4bit_quant_type='nf4',  # Utilise NormalFloat4 pour une meilleure précision
    bnb_4bit_compute_dtype=torch.float16  # Utilise float16 pour les calculs
  )
  return quantization_config

@measure_time
def _model_init():
    """
    Selon CUDA avec PyTorch, initialisera avec GPU CUDA ou CPU
    """
    if USE_QUANTIZATION:
        # https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
        # max_memory = {0: "10GiB", "cpu": "32GiB"}
        _model = AutoModelForCausalLM.from_pretrained(
            get_model_name_llm(),
            torch_dtype=torch.bfloat16,
            quantization_config=_getConfigQuantification(),
            # max_memory=max_memory,
            device_map='auto',  # répartit automatiquement entre CPU et GPU
            # llm_int8_enable_fp32_cpu_offload=True # uniquement en 8 bits
        )
        _device = torch.device(_cuda_or_cpu())
        return _model.to(_device)
    else:
        return AutoModelForCausalLM.from_pretrained(get_model_name_llm(),
                                                    torch_dtype=torch.float16
                                                    ).to(_cuda_or_cpu())


@measure_time
def generate_response_nlp(prompt, model):
    """Génération de la réponse Llama 2"""

    tokenizer = _get_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt").to(_cuda_or_cpu())

    # pour calcul de l'inférence en tokens / sec
    start_time = time.time()
    output = model.generate(**inputs, max_new_tokens=200)
    end_time = time.time()

    generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    inference_time = end_time - start_time
    tokens_per_sec = generated_tokens / inference_time if inference_time > 0 else float("inf")
    print(f"Vitesse inférence : {tokens_per_sec:.2f} tokens/sec")

    return tokenizer.decode(output[0], skip_special_tokens=True)

@measure_time
def search_responses(question):
    """
    Recherche des vecteurs dans ChromaDB et génération de réponse avec Llama 2
    """
    model, collection = _init_llm()

    embedding_model = get_embedding_model_chroma()
    query_embedding = embedding_model.embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    # print(f"Résultats trouvés : {results.similarity_search_with_score(question, 1)}")
    print(f"{results}")
    retrieved_texts = " ".join([doc for doc in results["documents"][0]])
    prompt = f"Extraits de ChromaDb trouvés : {retrieved_texts}\n\nQuestion : {question}\nRéponse :"

    # return generate_response_nlp(prompt, model)
    return prompt, model

if __name__ == '__main__':
    _init_logger()
    question = "Quels sont les points clés du document ?"
    prompt, model = search_responses(question)
    print(generate_response_nlp(prompt, model))
    # print(search_responses(question))
