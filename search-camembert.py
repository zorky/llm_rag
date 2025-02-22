from transformers import AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLM, AutoTokenizer
import chromadb
import torch

from utils import logger
from utils.model_embeddings import get_embedding_model, get_model_name
from utils.duration_decorator import measure_time

LOGGER = "raggy_logger"

CHROMA_DB = "./chroma_db3"
CHROMA_COLLECTION = "docs"

USE_GPU = True

has_cuda = torch.cuda.is_available()
print(f"Support GPU : {has_cuda}")  # Doit renvoyer True si CUDA est bien installé
if has_cuda:
    print(f"Version CUDA : {torch.version.cuda}")  # Doit afficher la version de CUDA utilisée

@measure_time
def chroma_load():
    """
    Charge les collections de vecteurs / embeddings de ChromaDB
    """
    _init_logger()
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB)
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
    return collection

# @measure_time
# def _generate_multi_pass(llm_model, llm_tokenizer, inputs):
#     """
#     Génération de la réponse en langage naturel, max_new_tokens est la limite du nombre de mots à rendre, plus elle est élevée, plus la réponse est longue
#     Evite de dépasser une réponse de 4096 tokens limite de Llama 2
#     """
#     response = ""
#     for _ in range(3):  # Générer en plusieurs passes
#         new_tokens = llm_model.generate(**inputs, max_new_tokens=200)
#         # new_tokens = llm_model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
#         new_text = llm_tokenizer.decode(new_tokens[0], skip_special_tokens=True)
#
#         response += new_text + " "
#         inputs = llm_tokenizer(response, return_tensors="pt", truncation=True).to(cgpu)  # Reprendre le texte
#     return response

@measure_time
def generative_response_llm(prompt):
    llm_name = "NousResearch/Llama-2-7b-chat-hf"  # Ou un autre modèle génératif adapté
    if has_cuda and USE_GPU:
        cgpu = "cuda"
    else:
        cgpu = "cpu"

    llm_model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16).to(cgpu)
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True).to(cgpu)

    # response = _generate_multi_pass(llm_model, llm_tokenizer, inputs)
    # Génération de la réponse en langage naturel, max_new_tokens est la limite, plus elle est élevée, plus la réponse est longue
    # Max 4096 tokens pour Llama 2
    output_tokens = llm_model.generate(**inputs, max_new_tokens=200)
    response = llm_tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return response

@measure_time
def search_response(question, collection):
    """
    Recherche des vecteurs dans ChromaDB et génération de réponse avec Llama 2
    """
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding], # optionnel si ChromaDb a été initialisé avec des embeddings pour une recherche sémantique
        n_results=3
    )
    retrieved_texts = "\n\n".join([doc for doc in results["documents"][0]])
    prompt = f"Réponds à la question en te basant sur ces documents : {retrieved_texts}\n\nQuestion : {question}\nRéponse :"

    return generative_response_llm(prompt)

def _init_logger():
    logger.setup_logging()
    log = logger.get_logger(LOGGER)
    return log

if __name__ == '__main__':
    _collection = chroma_load()
    question = "Quels sont les points clés du document ?"
    print(search_response(question, _collection))
    # question = input("Poser une question : ")
    # print(search_response(question))