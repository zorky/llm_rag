from langchain_huggingface import HuggingFaceEmbeddings

def get_model_name():
    """
    Modèle CamemBERT spécialisé en français : https://huggingface.co/dangvantuan/sentence-camembert-large
    """
    return "dangvantuan/sentence-camembert-large"

def get_embedding_model():
    model_name = get_model_name()
    return HuggingFaceEmbeddings(model_name=model_name)

