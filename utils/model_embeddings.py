from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model_chroma():
    """ Transformer pour les embeddings à stocker ou la recherche dans ChromaDB """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_model_name_llm():
    """ Charger Llama 2 pour générer une réponse humaine en langage naturel """
    return "NousResearch/Llama-2-7b-chat-hf"
