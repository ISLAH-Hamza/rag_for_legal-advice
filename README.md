
# Configuration et Installation de l'Environnement pour un Projet RAG en Python

Pour travailler sur un projet Python tel qu'un système de RAG (Retrieval-Augmented Generation) pour l'assistance juridique, il est essentiel d'avoir un environnement bien configuré. Ce guide vous aidera à installer et configurer l'environnement de développement en utilisant Python et/ou Anaconda.



## Étape 0 : Vérifier et Installer Python
Avant de commencer, assurez-vous d'avoir un interpréteur Python installé sur votre machine. Il existe deux principales options :

- **Installer Anaconda** : Recommandé si vous travaillez avec du data science ou du machine learning. [Télécharger Anaconda](https://www.anaconda.com/download)
- **Installer Python directement** : Option plus légère et flexible. [Télécharger Python.](https://www.python.org/downloads/)

Vérifiez si Python/anconda est déjà installé en exécutant la commande suivante dans un terminal :

```shell
python --version
```

Si Python est installé, vous verrez une sortie indiquant la version, par exemple : \texttt{Python 3.11.2}.
Si ce n'est pas le cas, installez Python en suivant l'un des liens ci-dessus.



## Étape 1 : Création d'un Environnement Virtuel
Il est recommandé de créer un environnement virtuel pour chaque projet afin d'éviter les conflits entre différentes versions de bibliothèques.

#### Création d'un Environnement avec Conda
Si vous utilisez Anaconda, créez un environnement dédié avec la commande suivante :
```shell
conda create -n mon_projet python=3.11
```

#### Activation de l'Environnement Conda
```shell
conda activate mon_projet
```

#### Création d'un Environnement Virtuel avec venv (sans Anaconda)
Si vous préférez utiliser Python sans Anaconda, utilisez **venv** pour créer un environnement virtuel :
```shell
python -m venv mon_projet_env
```

#### Activation de l'Environnement Python (venv)

**Sur Windows** :
```shell
mon_projet_env\Scripts\activate
```
**Sur macOS/Linux** :

```shell
source mon_projet_env/bin/activate
```


Après activation, vous verrez probablement le nom de votre environnement affiché entre parenthèses dans votre terminal.



## Étape 2 : Installation des Bibliothèques Nécessaires
Une fois l'environnement activé, installez les bibliothèques requises pour votre projet. Ces dépendances sont généralement listées dans un fichier **requirements.txt**.

#### Installation des Packages avec pip
Si vous avez un fichier \texttt{requirements.txt}, installez toutes les dépendances en une seule commande :
```shell
pip install -r requirements.txt
```

#### Vérification des Installations
Après l'installation, vérifiez que les bibliothèques ont bien été installées en listant les packages disponibles :
```shell
pip list
```



## Étape 3 : Vérifier et Tester l'Environnement
Pour s'assurer que tout fonctionne correctement, vous pouvez exécuter une simple commande Python dans votre environnement activé :
```Python
import sys
print(sys.version)

```

Si la version de Python affichée correspond à celle de votre environnement, alors tout est bien configuré.

Vous avez maintenant un environnement Python propre et bien configuré pour travailler sur votre projet RAG pour l'assistance juridique. Il ne vous reste plus qu'à commencer à développer votre application !



## Étape 4 : Création du Projet
Nous allons maintenant créer un dossier pour notre projet sous le nom legal_rag.

1. Ajouter une Variable d'Environnement
Pour ce projet, nous avons besoin d'utiliser une clé API provenant d'OpenAI. Pour cela, nous allons créer un fichier .env et y ajouter notre clé API.

2. Ajouter un Dossier de Données
Nous allons également ajouter un dossier data où nous stockerons les données nécessaires au projet.




# RAG et base de donner vecotriel


## Indexation des Documents

L'indexation est une étape essentielle dans la construction d'un système RAG (Retrieval-Augmented Generation).
Elle consiste à transformer les documents sources en embeddings afin qu'ils puissent être stockés et recherchés efficacement dans une base de données vectorielle.
### Chargement des Documents PDF

Nous allons commencer par téléverser un dossier contenant des fichiers PDF et les manipuler avec la bibliothèque PyPDF de LangChain.
Exemple d'utilisation de PyPDF pour extraire du texte

```Python

from langchain.document_loaders import PyPDFLoader

# Charger un document PDF
loader = PyPDFLoader("chemin/vers/le/document.pdf")
documents = loader.load()

# Afficher le contenu extrait
documents[:2]  # Affiche les deux premiers segments extraits

```
### Segmentation des Documents en Chunks
Une fois le texte extrait, nous devons le diviser en morceaux (chunks) pour faciliter l'indexation et la recherche.


Méthodes de Chunking dans LangChain:

- RecursiveCharacterTextSplitter : Divise le texte en utilisant des caractères spécifiques comme limite.

- CharacterTextSplitter : Utilise un caractère (ex. '\n' ou ' ') pour scinder le texte.

- TokenTextSplitter : Divise le texte en fonction du nombre de tokens.

- NLTKTextSplitter : Utilise la bibliothèque NLTK pour le traitement du langage naturel.

- SentenceTransformersTextSplitter : Utilise des modèles de transformer pour diviser intelligemment le texte.


Exemple d'utilisation de RecursiveCharacterTextSplitter
```Python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Nombre de caractères par chunk
    chunk_overlap=50  # Chevauchement entre les chunks
)

chunks = text_splitter.split_documents(documents)
print(f"Nombre de chunks générés: {len(chunks)}")

```

Stockage des Chunks dans une Base de Données Vectorielle (ChromaDB)

Nous allons stocker ces chunks sous forme de vecteurs dans ChromaDB.
```Python
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Initialiser ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Convertir les chunks en embeddings et les stocker
embedding_function = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embedding_function, client=chroma_client)
```

## Récupération des Informations (Retrieval)
Une fois les embeddings stockés, nous pouvons créer un objet retriever pour rechercher les documents les plus pertinents.

Création d'un Retriever avec ChromaDB

```Python
    retriever = vector_store.as_retriever()
```

Exemples d'Utilisation de la Récupération d'Informations

```Python
query = "Quels sont les règlements juridiques pour les contrats de travail ?"
retrieved_docs = retriever.get_relevant_documents(query)

# Afficher les résultats
for doc in retrieved_docs:
    print(doc.page_content)
```

## Génération de Réponses Basées sur les Documents Récupérés
Après avoir récupéré les documents pertinents, nous pouvons les utiliser pour enrichir une réponse générée par un modèle de langage.

Example:
```Python
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

llm = OpenAI(model_name="gpt-4", temperature=0.5)
qa_chain = RetrievalQA(llm=llm, retriever=retriever)

query = "Quels sont les droits d'un employé en cas de licenciement ?"
response = qa_chain.run(query)
print(response)
```

## Limitations de cette Approche Naïve et Améliorations

Bien que cette approche de RAG fonctionne, elle présente plusieurs limitations :

Recherches basées uniquement sur la similarité vectorielle : Le modèle peut retourner des résultats pertinents en termes de distance dans l'espace des embeddings, mais pas forcément en fonction de la pertinence sémantique exacte.

Manque de pondération des résultats : Certains documents plus pertinents peuvent être noyés dans la masse.

Absence de fusion des résultats multi-sources : Lorsqu'un même sujet est traité dans plusieurs documents, l'agrégation d'information est limitée.

Manque de post-traitement intelligent : Les résultats bruts sont retournés sans réorganisation ou hiérarchisation avancée.

Améliorations avec des Techniques Avancées de RAG

Pour surmonter ces limitations, on peut utiliser des techniques avancées comme Reciprocal Rank Fusion (RRF) et d'autres méthodes de re-ranking :

1. Reciprocal Rank Fusion (RRF) :

Fusionne les résultats de plusieurs requêtes ou méthodes de recherche.

Améliore la pertinence en combinant plusieurs scores de recherche.

Extrait les meilleurs résultats en donnant une pondération plus équilibrée.

Implémentation de RRF avec ChromaDB

```Python
def reciprocal_rank_fusion(results_list, k=60):
    fused_scores = {}
    for rank, doc in enumerate(results_list, start=1):
        fused_scores[doc] = fused_scores.get(doc, 0) + 1 / (k + rank)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

2. Re-ranking avec un modèle Transformer :

Utiliser un modèle comme ColBERT pour améliorer la qualité des résultats récupérés.

Appliquer un reranking basé sur l'attention multi-head.

3. Fusion Multi-Sources :

Enrichir les résultats en combinant plusieurs sources de données externes.

Appliquer une pondération dynamique en fonction du contexte de la requête.

Avec ces techniques, on peut significativement améliorer la pertinence des résultats retournés et optimiser la performance du système RAG pour une meilleure assistance à la prise de décision