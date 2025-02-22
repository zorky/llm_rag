# LLM & RAG LLAMA 2 expérimentation

## Raggy Nano RAG

Expérimentation de création d'un nano RAG puis recherche dedans

## Prérequis

### HuggingFace

HuggingFace, un hub, est utilisé pour le modèle de langage et le modèle de recherche.

** /!\ attention, HuggingFace va puller les modèles utilisés et les stocker en local, cela peut représenter plusieurs Go ! /!\ **

Un compte est nécessaire afin de pouvoir _puller_ les modèles : https://huggingface.co/ 

Puis un token à initialiser : https://huggingface.co/settings/tokens et se logger sur HF via le cli : https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login

### Initialisation venv

Installe pytorch avec le support GPU CUDA

```bash
$ virtualenv env
$ source env/Scripts/activate
$ pip install -r requirements.txt
```

## Etapes

### 1- Constitution base vectorielle avec un modèle opensource : index-doc.py

- PDF : extraction en "chunks" de texte (extraits)
- BDD : stockage des extraits dans ChromaDb placé dans le répertoire `chroma_db`

Indexation d'un document PDF : 25 s

```bash
$ python index-doc.py # lit et index document.pdf
```

### 2- Recherche dans la base ChromaDb sous forme de questions / réponses avec modèle de réponse Llama 2 : search-doc.py

```bash
$ python search-doc.py # recherche dans la base indexée
```

Temps d'exécution :

- initialisation modèle : 50 s
- Chroma chargement : 1,5 s
- Recherche sur la phrase : 350 s (5 min 50 s)