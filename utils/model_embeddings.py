from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model_chroma():
    """
    Transformer pour les embeddings à stocker ou la recherche dans ChromaDB
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_model_name_llm():
    """
    Modèle quantifié Mistral-7 7B : https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
    via llama.cpp https://github.com/ggml-org/llama.cpp
    """
    # return "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    # return "TheBloke/Mistral-7B-GPTQ"
    # return "mistralai/Mistral-7B-Instruct-v0.2"
    return "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
