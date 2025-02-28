# LLM & RAG Mistral 7B Instruct expérimentation

## Raggy Nano RAG

Expérimentation de création d'un nano RAG à partir d'un PDF puis question / réponse (inférence)

## Prérequis

### NVidia et CUDA

PyTorch est installé avec CUDA (cf. requirements.txt).

Sous Windows, pour la prise en compte de CUDA, cela demandera peut-être d'installer le [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local) 

Vérifier que CUDA est bien pris en charge par PyTorch, après installation des packages (cf. Initialisation venv plus bas) :

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

Les modèles utilisés (Mistral) peuvent demander une autorisation pour les conditions d'utilisation, directement sur les espaces concernés :

- Mistal 7 Instruct : https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

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

Model sentence-transformers/all-MiniLM-L6-v2 (91.6M, used 2 hours ago)
  ○ fa97f6e7: main # modified 4 weeks ago

Model NousResearch/Llama-2-7b-chat-hf (13.5G, used 2 hours ago)
  ○ 351844e7: main # modified 1 week ago

Model dangvantuan/sentence-camembert-large (1.3G, used 17 minutes ago)
  ○ 1f111fd0: main # modified 6 days ago

Model mistralai/Mistral-7B-Instruct-v0.3 (14.5G, used 17 minutes ago)
  ○ e0bc86c2: main # modified 42 minutes ago

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
$ python search-mistal.py # recherche dans la base indexée et inférence de la réponse
```

Temps d'exécution :

- initialisation modèle : 50 s
- Chroma chargement : 1,5 s
- Recherche sur la phrase "Quels sont les points clés du document ?": 
  - GPU : 350 s (5 min 50 s) mais utilise aussi le CPU car la carte graphique est trop limitée (2 Go VRAM)
    
-----------------

### Résultat de la question

**Question :** Quels sont les points clés du document ?

**Extraits de ChromaDb trouvés via les vecteurs / embeddings**

```
surnoms blessants, mises à l’écart…).

- L’intention : Souvent les harceleurs disent qu’ils agissent « juste
pour rire ». Mais derrière cette prétendue rigolade, ils ont surtout envie
de montrer leur pouvoir ou d’exercer leur force ! Parfois ils veulent
juste faire souffrir quelqu’un, sans aucune raison.

- La répétition : Le harcèlement, c’est la répétition d’une multitude de
petites attaques. Comme une goutte d’eau qui fuit du robinet et ne C’est quoi le harcèlement ?

CA FAIT MAL !
Le harcèlement à l’école, ce sont des violences répétées par un ou plusieurs
élèves contre un/une autre camarade. Le ou les agresseurs agissent pour le
dominer, le blesser et l’exclure. On parle de harcèlement quand on retrouve
ces 4 éléments :
- La violence : Ce sont parfois des actes très graves (insultes, coups,
vols, etc.) mais aussi des gestes qui paraissent plus banals (moqueries,
surnoms blessants, mises à l’écart…). petites attaques. Comme une goutte d’eau qui fuit du robinet et ne
s’arrête jamais. Séparément, ces actes ne paraissent pas graves. Mais
répétés, ils blessent. Avec le temps, ils deviennent de plus en plus
violents.

- L’isolement : C’est à la fois une cause et une conséquence. Un/Une
enfant peut être harcelé(e) parce qu’il/elle est différent(e) des autres
(trop grand(e), trop bizarre, trop machin, trop truc…), mais aussi parce
```

**Génération de la réponse par Llama 2** qui crée un résumé, qui peut être différent à chaque exécution

```
Les points clés du document sont les suivants :

1. L'intention des harceleurs : Ils agissent souvent pour montrer leur pouvoir ou exercer leur force, ou bien pour simplement faire souffrir quelqu'un sans aucune raison.
2. La répétition : Le harcèlement est la répétition d'une multitude de petites attaques.
3. La violence : Les actes peuvent être très graves (insultes, coups, vols, etc.) ou bien des gestes qui paraissent plus banals (moqueries, surnoms blessants, mises à l'écart...).
4. L'isolement : C'est à la fois une cause et une conséquence. Un/Une enfant peut être harcelé(e) parce qu'il/elle est différent(e) des autres, mais
```

```
Les points clés du document sont :

1. L'intention des harceleurs : Ils agissent souvent pour montrer leur pouvoir ou exercer leur force, ou bien pour simplement faire souffrir quelqu'un sans aucune raison.
2. La répétition : Le harcèlement est la répétition d'une multitude de petites attaques.
3. La violence : Les actes peuvent être très graves (insultes, coups, vols, etc.) ou bien des gestes qui paraissent plus banals (moqueries, surnoms blessants, mises à l'écart...).
4. L'isolement : C'est à la fois une cause et une conséquence. Un/Une enfant peut être harcelé(e) parce qu'il/elle est différent(e) des autres, mais
```

### Schéma des traitements

![schema-scripts.png](schema-scripts.png)

source : https://excalidraw.com/#json=CBUbyYpTHmf3KcoJeFdT6,ZNnQyrbAt-Lv87FAlxuKrg  
