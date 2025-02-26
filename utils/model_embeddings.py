from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model_chroma():
    """
    Transformer pour les embeddings à stocker ou la recherche dans ChromaDB
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_model_name_llm():
    """ 
    Charger Llama 2 pour générer une réponse humaine en langage naturel 
    NousResearch : https://huggingface.co/NousResearch/Llama-2-7b-chat-hf
    meta (demandera une autorisation) : https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
    """
    return "NousResearch/Llama-2-7b-chat-hf"
