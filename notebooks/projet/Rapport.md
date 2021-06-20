-------------------------------------------------------------------
**TD DSA 2021 de Antoine Ly   -   rapport de Fabien Faivre**
-------------------------     -------------------------------------

-------------------------------------------------------------------
# Instructions

L'objectif est de vous faire approfondir une notion de machine learning aux travers d'une compétition Kaggle. Kaggle est aujourd'hui un site de compétition incontournable dans le monde du machine learning. Même si la plateforme n'est pas représentative des enjeux opérationnels, elle est néanmoins représentative des sujets d'attention de la communauté scientifique et demeure un bon outil d'apprentissage est de partage.


Le projet consiste donc à utiliser le challenge [Tweet sentiment extraction](https://www.kaggle.com/c/tweet-sentiment-extraction/overview/description) à des fins académiques.


### Les données

Les données sont celles proposées par le challenge. Elles composent de deux fichiers:

* `train.csv` ce document comportent les données à utiliser pour calibrer votre modèle. Il comporte toutes les colonnes
* `test.csv` ce document n'est utilisé **que** pour évaluer la performance finale de votre modèle. En aucun cas il ne peut être utilisé pour fine-tuner ou calibrer votre modèle. Il simule les données qui ne sont normalement JAMAIS accessible sur Kaggle (ni dans la vraie vie). à considérer comme un nouvel échantillon.

### Le challenge

Prédire la colonne `sentiment` à partir de la colonne `text`.

### La métrique d'évaluation

On utilisera un score F1 à l'aide de la fonction implémentée dans `scikit-learn`

    from sklearn.metrics import f1_score
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    f1_score(y_true, y_pred, average='macro')


### Labels à utiliser pour la colonne `sentiment`

Vous devrez retraiter la colonne `sentiment` en utilisant les remplacements suivants:

    "neutral"  ->  0
    "negative" -> -1
    "positive" ->  1

### Informations pratiques sur le rendu et la notation.

L'objectif est de se familiariser avec les techniques de text-mining à des fins de classification de sentiments d'un texte. La notation se décomposera en deux parties:

#### Notation

* Votre méthodologie et votre approche (12 points) : cette partie doit mettre en avant la motivation des différents retraitements que vous avez appliquez, votre effort de comprendre les implémentations des pacakges que vous aurez utilisés ainsi que le bon sens que leurs utilisations transcrit.
* La performance finale et méthodologie (4 + 4 = 8 points) : cette notation sera relative au groupe. 3 tentatives d'algorithmes/preprocessing différents permettrons de garantir 4 points sur les 8. Les 3 premiers du classement (du groupe 2020) calculé à l'aide du score F1 sur la base de test atteigneront 4 points supplémentaires. Le reste du barême relatif au classement sera dégressif de façon linéaire par palier: les derniers obtenant 1 point minimum.

#### Rendu

Le rendu se fera sous la forme d'un court rapport (max 5 pages). Ce dernier peut se faire sous la forme d'un notebook (html ou pdf) ou d'un rapport traditionnel (word, pdf). Il doit mettre en avant la méthodologie employée, les difficultés rencontrées ainsi que les différents apprentissages.


Le projet sera à rendre lors de la séance de **Juillet 2021** de restitution.

### Language de programmation

Il est fortement recommandé d'effectuer le projet en python, mais ceci n'est pas obligatoire.

Bibliographie:

https://www.scor.com/fr/articles-experts/accroitre-vitesse-et-precision-grace-lexploration-de-texte-et-au-traitement

-------------------------------------------------------------------

# Description du projet et de la stratégie a priori

L'approche déployée consiste à analyser des tweets en langue anglaise et de prédire les sentiments qu'ils portent : `{negative: -1, neutral: 0, positive: 1}`

Dans cet exercice, la langue anglaise est un facteur facilitant dans la mesure où beaucoup de modèles préentrainés existent dans cette langue.

La difficulté dans cet exercice provient de sa source : les tweets.
Les approches classiques reposent sur :
- le passage en minuscule, or dans les tweets, l'utilisation de mots en **majuscules** est un marqueur d'une **émotion forte**
- l'utilisation de structures linguisitiques relativement correctes augmentées par la lemmatisation / tokenisation. 
Or, les mots utilisés dans les tweets font l'objet de nombreuses **fautes d'orthographes ou d'abbréviations** (ex `thx`)
- les marqueurs de ponctuation sont usuellement retirés, or ici, ils peuvent être utilisés comme **smiley** `;-)` ou pour marquer une **émotion forte** `!!!`
- l'**humour** et les **euphémismes** sont très présents sur tweeter, or les modèles ont beaucoup de mal à distinguer ces cas qui nécessitent une compréhension contextuelle.

En complément au sujet du TD lui-même, celui-ci a été l'occasion de monter en compétence avec les (je l'espère) bonnes pratiques de codage et l'utilisation de techniques de MLOps.

Le code de ce projet a été organisé en s'appuyant sur le framework open source [**orbyter**](https://github.com/manifoldai/orbyter-cookiecutter) de la société [Manifold.ai](https://www.manifold.ai/project-orbyter). Ce framework pousse à la standardisation de la structure du code, via l'utilisation de `cookiecutter` et promeut un développement dans un environnement dockerisé dès le départ : 

![structure](https://www.manifold.ai/hubfs/Torus.png) 

La logique de développement pronée est disponible [ici](https://cdn2.hubspot.net/hubfs/4584542/Conference%20Slides/2019StrataNY_EfficientMLengineering.pdf)

Dans cette approche, le code est développé dans un environnement dockerisé, afin de faciliter la reproductibilité, ainsi que le déploiement dans le cloud au besoin.
Les expérimentations sont stockées dans un serveur MLFlow pour archivage, comparaison et utilisation.

Plusieurs modifications ont dû être apportées aux paramètres du `docker-compose` pour permettre un accès aux ressources GPU depuis le docker.

Le code a été versionné et est disponible ici [github](https://github.com/Fabien-DS/DSA_Sentiment)

-------------------------------------------------------------------

# Chargement des packages nécessaires

Cette section technique, sert uniquement à montrer les outils utilisés

```python

#Temps et fichiers
import os
import warnings
import time
from datetime import timedelta

#Manipulation de données
import pandas as pd
import numpy as np


# Text
from collections import Counter
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.util import ngrams

from textblob import TextBlob
import string
import re
import spacy 
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

#Modélisation
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


#Evaluation
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score


#Visualisation
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

#Tracking d'expérience
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

```
Les fonctions principales ont été factorisées dans un package spécifique au projet et utilisées dans les notebooks à l'aide de l'instruction suivante :

```python
#Cette cellule permet d'appeler la version packagée du projet et d'en assurer le reload avant appel des fonctions
%load_ext autoreload
%autoreload 2
```

