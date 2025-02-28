from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model_chroma():
    """
    Transformer pour les embeddings à stocker ou la recherche dans ChromaDB
    https://huggingface.co/dangvantuan/sentence-camembert-large
    """
    return HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-large")

def get_model_name_llm():
    """ 
    Charger Mistral 7B Instruct pour générer une réponse humaine en langage naturel
    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 : meta : https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/config.json
    https://huggingface.co/mistralai/Mistral-7B-v0.3
    """
    # return "mistralai/Mistral-7B-v0.3" # segmentation fault :(
    return "mistralai/Mistral-7B-Instruct-v0.3"

