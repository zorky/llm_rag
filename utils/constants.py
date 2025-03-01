CHROMA_DB = "./chroma_db0_embeddings"
CHROMA_COLLECTION = "docs"
CHUNK_SIZE=500
CHUNK_OVERLAP=100
PROMPT_TEMPLATE = """
Répondez en français à la question en basant uniquement sur le contexte suivant :

{context}

Répondez à la question suivante en basant sur le contexte ci-dessus: {question}
"""

LOGGER = "raggy_logger"