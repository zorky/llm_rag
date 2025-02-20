# LLM & RAG LLAMA 2

## Prérequis

### HuggingFace

Un compte est nécessaire afin de pouvoir puller les modèles : https://huggingface.co/ 

Puis un token à initialiser : https://huggingface.co/settings/tokens et certainement se logger sur HF via le cli : https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login

### Initialisation venv

Installe pytorch avec le support GPU CUDA

```bash
$ virtualenv env
$ source env/Scripts/activate
$ pip install -r requirements.txt
```

## Etapes

### 1- Constitution base vectorielle : index-doc.py

- PDF : extraction en "chunks" de texte (extraits)
- BDD : stockage des extraits dans ChromaDb placé dans le répertoire `chroma_db`

```bash
$ python index-doc.py # lit et index document.pdf
```

### 2- Recherche dans la base ChromaDb sous forme de questions / réponses

```bash
$ python search-doc.py # recherche dans la base indexée
```
