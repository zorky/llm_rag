# LLM & RAG CamemBERT & LLAMA 2 expérimentation

## Raggy Nano RAG

Expérimentation de création d'un nano RAG puis recherche dedans

## Prérequis

### HuggingFace

HuggingFace, un hub, est utilisé pour le modèle de langage et le modèle de recherche.

#### Compte HF

Un compte est nécessaire afin de pouvoir _puller_ les modèles : https://huggingface.co/ 

Puis un token à initialiser : https://huggingface.co/settings/tokens et se logger sur HF via le cli : https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login

#### Cli HF

** /!\ attention, HuggingFace va puller les modèles utilisés et les stocker en local, cela peut représenter plusieurs Go ! /!\ **

Le CLI HF permet de vide son cache de modèles stockés sur le disque 
```bash
$ pip install -U "huggingface_hub[cli]"
```

Il demandera quels modèles à supprimer en sélectionnant le modèle à supprimer

```bash
$ huggingface-cli delete-cache

? Select revisions to delete: 0 revisions selected counting for 0.0.
❯ ○ None of the following (if selected, nothing will be deleted).

Model meta-llama/Llama-2-7b-chat-hf (13.5G, used 2 days ago)
  ○ f5db02db: main # modified 2 weeks ago

Model sentence-transformers/all-mpnet-base-v2 (438.7M, used 2 days ago)
  ○ 9a322596: main # modified 4 weeks ago

Model NousResearch/Llama-2-7b-chat-hf (13.5G, used 16 minutes ago)
  ○ 351844e7: main # modified 3 days ago

Model sentence-transformers/all-MiniLM-L6-v2 (91.6M, used 15 minutes ago)
  ○ fa97f6e7: main # modified 3 weeks ago

```

### Initialisation venv

Installe pytorch avec le support GPU CUDA

```bash
$ virtualenv env
$ source env/Scripts/activate
$ pip install -r requirements.txt
```

## Etapes

### 1- Constitution base vectorielle avec un modèle opensource : index-doc.py

Avec CamemBERT.

- PDF : extraction en "chunks" de texte (extraits)
- BDD : stockage des extraits dans ChromaDb placé dans le répertoire `chroma_db`

Indexation d'un document PDF : 25 s

```bash
$ python index-doc.py # lit et index document.pdf
```

### 2- Recherche dans la base ChromaDb sous forme de questions / réponses

CamemBERT est utilisé pour la recherche dans Chroma.
Llama 2 pour générer la réponse en langage naturel.

```bash
$ python search-camembert.py # recherche dans la base indexée
```

Temps d'exécution :

- initialisation modèle : 50 s
- Chroma chargement : 1,5 s
- Recherche sur la phrase : 350 s (5 min 50 s)