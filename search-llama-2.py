from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
import torch

import logger
from utils import measure_time

LOGGER = "raggy_logger"

has_cuda = torch.cuda.is_available()
print(f"Support GPU : {has_cuda}")  # Doit renvoyer True si CUDA est bien installé
if has_cuda:
    print(f"Version CUDA : {torch.version.cuda}")  # Doit afficher la version de CUDA utilisée

# Charger Llama 2 (modèle léger pour test)
model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

@measure_time
def chroma_init():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="docs")
    return collection

@measure_time
def model_init():
    if has_cuda:
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
    else:
        # force le CPU vs GPU CUDA
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cpu")
    return model

@measure_time
def generate_response(question, model, collection):
    # Recherche dans ChromaDB
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    retrieved_texts = " ".join([doc for doc in results["documents"][0]])
    prompt = f"Voici des informations utiles : {retrieved_texts}\n\nQuestion : {question}\nRéponse :"

    # Génération avec Llama 2
    if has_cuda:
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

if __name__ == '__main__':
    _init_logger()
    _model = model_init()
    _collection = chroma_init()
    question = "Quels sont les points clés du document ?"
    print(generate_response(question, _model, _collection))
    # question = input("Poser une question : ")
    # print(generate_response(question))