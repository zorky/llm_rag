# LLM & RAG expérimentation

## Tentative d'accélération de l'inférence de la réponse

Tentative d'autres méthodes pour accéler l'inférence de la réponssse : 

- GGUF : modèle ultra-quantifé par llama-cpp mais empêche l'utilisation des Transformers : KO
- GPTQ : optimisé pour le GPU : aucun gain de performance, même pire : KO

Cette branche n'est pas fonctionnelle, aucun apport d'amélioration de performance.

## Prérequis

### NVidia et CUDA

PyTorch est installé avec CUDA (cf. requirements.txt).

Sous Windows, pour la prise en compte de CUDA, cela demandera peut-être d'installer le [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local) 

Vérifier que CUDA est bien pris en charge par PyTorch, après installation des packages (cf. Initialisation venv plus bas) :
[README.md](README.md)
```bash
$ python
>>> import torch
>>> torch.cuda.is_available()
True
>>> exit()
```
 

### HuggingFace

HuggingFace, un hub, est utilisé pour le modèle de langage et le modèle de recherche.

#### Compte Hugging Face & autorisation d'utilisation des modèles

Un compte Hugging Face et un token vont être nécessaire, et également une demande d'autorisation d'utilisation sur les espaces des modèles

- Se créer un compte : https://huggingface.co/
- Initialiser un token : https://huggingface.co/settings/tokens
- Se logger sur HF via le cli avec le token précédemment généré (https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login)

```bash
$ huggingface-cli login
Enter your token (input will not be visible):
```

Savoir si on est connecté à HG avec le cli (cf. ci-après)

```bash
$ huggingface-cli whoami
Not logged in
```

Les modèles utilisés (Llama 2) demandent une autorisation pour les conditions d'utilisation, directement sur les espaces concernés :

- NousResearch : https://huggingface.co/NousResearch/Llama-2-7b-chat-hf
- Meta : https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

#### Cli Hugging Face

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

- PDF : extraction en "chunks" de texte (extraits)
- BDD : stockage des extraits dans ChromaDb placé dans le répertoire `chroma_db`

Indexation d'un document PDF : 25 s

```bash
$ python index-doc.py # lit et index document.pdf
```

### 2- Recherche dans la base ChromaDb sous forme de questions / réponses avec modèle de réponse Llama 2 : search-doc.py

```bash
$ python search-response.py # recherche dans la base indexée
```
