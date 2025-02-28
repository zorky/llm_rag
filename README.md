# Expérimentations RAG & LLM locale

## Raggy Nano RAG

Expérimentation de création d'un nano RAG puis recherche / inférence dedans sur une question

**Branches :**

**0-** `raggy-0.0` : Pas d'indexation avec vecteurs et pas de génération de réponse via un LLM, uniquement recherche textuel dans la BDD Chroma https://github.com/zorky/llm_rag/tree/raggy-0.0 

**1-** `raggy-0.1` : Indexation des embeddings des documents avec le modèle sentence-transformers/all-MiniLM-L6-v2 et génération de réponse avec le **Llama 2** : https://github.com/zorky/llm_rag/tree/raggy-0.1

**2-** `raggy-0.2` : Indexation des embeddings des documents avec **CamemBERT**, modèle spécialisé en français (pour embeddings BDD Chroma) et génération de réponse avec **Llama 2** : https://github.com/zorky/llm_rag/tree/raggy-0.2

**3-** `raggy-0.3` : essais avec modèles pour accélérer l'inférence en GGUF et GPTQ : non fonctionnel et aucun gain

**4-** `raggy-0.4` : Indexation des embeddings des documents avec **CamemBERT** (pour embeddings BDD Chroma) et génération de réponse avec **Mistral 7B** : https://github.com/zorky/llm_rag/tree/raggy-0.4/
