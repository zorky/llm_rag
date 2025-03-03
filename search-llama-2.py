from transformers import AutoTokenizer
import time

from utils.logger import init_logger
from utils.model_embeddings import get_embedding_model_chroma, get_model_name_llm, calculate_inference_tokens_time, \
    model_inference_init, cuda_or_cpu
from utils.measure_time_decorator import measure_time
from utils.vectors_db import chroma_load

@measure_time
def generate_response_nlp(prompt):
    """Génération de la réponse Llama 2"""
    model = model_inference_init()
    tokenizer = AutoTokenizer.from_pretrained(get_model_name_llm())

    inputs = tokenizer(prompt, return_tensors="pt").to(cuda_or_cpu())

    # pour calcul de l'inférence en tokens / sec
    start_time = time.time()
    output = model.generate(**inputs, max_new_tokens=200)
    end_time = time.time()

    generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]

    calculate_inference_tokens_time(generated_tokens, start_time, end_time)

    return tokenizer.decode(output[0], skip_special_tokens=True)

@measure_time
def search_responses(question, max_results=3):
    """
    Recherche sémantique
    des vecteurs dans ChromaDB et génération de réponse avec Llama 2
    """
    collection = chroma_load()

    embedding_model_db = get_embedding_model_chroma()
    query_embedding = embedding_model_db.embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max_results
    )
    print(f"{results}")
    retrieved_texts = " ".join([doc for doc in results["documents"][0]])
    prompt = f"Extraits de ChromaDb trouvés : {retrieved_texts}\n\nQuestion : {question}\nRéponse :"

    return prompt

if __name__ == '__main__':
    init_logger()
    question = "Quels sont les points clés du document ?"
    prompt = search_responses(question)
    print(generate_response_nlp(prompt))
