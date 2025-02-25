# LLM & RAG LLAMA 2 expérimentation

## Raggy Nano RAG

Expérimentation de création d'un nano RAG puis recherche dedans sur une question

Branches :

**0-** `raggy-0.0` : Pas de génération de réponse via un LLM, uniquement recherche dans la BDD vectorielle Chroma https://github.com/zorky/llm_rag/tree/raggy-0.0 

**1-** `raggy-0.1` : Indexation des embeddings des documents avec le modèle sentence-transformers/all-MiniLM-L6-v2 et génération de réponse avec le LLM Llama 2 : https://github.com/zorky/llm_rag/tree/raggy-0.1

**2-** `raggy-0.2` : Indexation des embeddings des documents avec CamemBERT (pour embeddings BDD Chroma) et génération de réponse avec Llama 2 : https://github.com/zorky/llm_rag/tree/raggy-0.2

