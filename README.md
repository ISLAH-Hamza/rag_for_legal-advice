
# Configuration et Installation de l'Environnement pour un Projet RAG en Python

Pour travailler sur un projet Python tel qu'un système de RAG (Retrieval-Augmented Generation) pour l'assistance juridique, il est essentiel d'avoir un environnement bien configuré. Ce guide vous aidera à installer et configurer l'environnement de développement en utilisant Python et/ou Anaconda.

## Étape 0 : Vérifier et Installer Python
Avant de commencer, assurez-vous d'avoir un interpréteur Python installé sur votre machine. Il existe deux principales options :

**Installer Anaconda** : Recommandé si vous travaillez avec du data science ou du machine learning. [Télécharger Anaconda](https://www.anaconda.com/download)
**Installer Python directement** : Option plus légère et flexible. [Télécharger Python.](https://www.python.org/downloads/)

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