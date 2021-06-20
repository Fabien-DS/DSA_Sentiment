#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
# **TD DSA 2021 de Antoine Ly   -   rapport de Fabien Faivre**
# -------------------------     -------------------------------------

# # Modélisation

# ## Setup

# In[23]:


get_ipython().system('pip install textblob')


# In[24]:


get_ipython().system('pip install emot')


# In[25]:


get_ipython().system('pip install wordcloud')


# In[26]:


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
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


#Evaluation
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score


#Visualisation
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

#Tracking d'expérience
import mlflow
import mlflow.sklearn


# In[27]:


#Cellule strictement technique qui permet de sauver les exigences pour recréer au besoin l'image docker du projet
get_ipython().system('pip freeze > /mnt/docker/requirements.txt')


# ### Utilisation du package

# Durent ce projet, certaines parties du code ont été re packagées dans un package propre au projet afin de factliter la lecture du core et permettre la réutilisabilité des développements

# In[28]:


#Cette cellule permet d'appeler la version packagée du projet et d'en assurer le reload avant appel des fonctions
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[29]:


from dsa_sentiment.scripts.make_dataset import load_data
from dsa_sentiment.scripts.evaluate import eval_metrics
from dsa_sentiment.scripts.make_dataset import Preprocess_StrLower, Preprocess_transform_target


# ### Configuration de l'experiment MLFlow

# [MLFlow](https://mlflow.org/) sera utilisé comme outil de suivi et de stockage des expérimentatiosn réalisées

# In[30]:


mlflow.tracking.get_tracking_uri()


# ### Chargement des données

# In[50]:


# On Importe les données

#df
df_train=pd.read_parquet('/mnt/data/interim/df_train.gzip')
df_val=pd.read_parquet('/mnt/data/interim/df_val.gzip')
df_test=pd.read_parquet('/mnt/data/interim/df_test.gzip')

#X
X_train=pd.read_parquet('/mnt/data/interim/X_train.gzip')
X_val=pd.read_parquet('/mnt/data/interim/X_val.gzip')
X_test=pd.read_parquet('/mnt/data/interim/X_test.gzip')

X_train_prepro=pd.read_parquet('/mnt/data/interim/X_train_prepro.gzip')
X_val_prepro=pd.read_parquet('/mnt/data/interim/X_val_prepro.gzip')
X_test_prepro=pd.read_parquet('/mnt/data/interim/X_test_prepro.gzip')

#y
y_train=pd.read_parquet('/mnt/data/interim/y_train.gzip')
y_val=pd.read_parquet('/mnt/data/interim/y_val.gzip')
y_test=pd.read_parquet('/mnt/data/interim/y_test.gzip')


# ## Modélisation

# ### Création du code générique 

# On commence par définir une fonction générique qui sera en capacité d'ajuster, optimiser et logger dans MLFlow les résultats de pipelines qui seront produits pour chaque essai

# Le mode de fonctionnement souhaité consiste à 
# 
# 1- définir un pipeline au sens de sklearn
# 
# 2- utiliser une fonction générique pour ajuster le pipeline (éventuellement en optimisant les paramètres) et en stocker le résultat dans MLFlow

# #### Préalables : création des fonctions de résultat souhaitées

# La première étape consiste à construire une fonction générique qui calculera **les scores du pipeline que nous souhaitons suivre**.
# Dans le cas présent comme l'exercice de classification est multiclasse, nous sommes intéressés par les `f1`, `precision` et `recall` calculés avec l'option `macro` qui réalise une moyenne des résultats obtenus par classe.

# In[32]:


def score_estimator(
    estimator, X_train, X_test, df_train, df_test, target_col
):
    
    """
    Evalue un pipeline sur le jeu de train et test avec plusieurs métriques
    
    Ici les métriques utilisées sont :
    - f1 macro
    - precision macro
    - recall macro
    
    INPUTS :
        - estimator : un pipeline
        - X_train, X_test, df_train, df_test : les DataFrames contenant les jeux de données et test
        - target_col : le nom de la colonne cible dans les df
        
    OUTPUTS :
        - un DataFrame avec les métriques calculées sur les jeux de train et test fournis
    
    
    """

    metrics = [
        ("f1_macro", f1_score),   
        ("precision_macro", precision_score),
        ("recall_macro", recall_score),
        
    ]
    
    res = []
    for subset_label, X, df in [
        ("train", X_train, df_train),
        ("test", X_test, df_test),
    ]:
        y = df[target_col]
        y_pred = estimator.predict(X)
        for score_label, metric in metrics:
            score = metric(y, y_pred, average='macro')
            res.append(
                {"subset": subset_label, "metric": score_label, "score": score}
            )

    res = (
        pd.DataFrame(res)
        .set_index(["metric", "subset"])
        .score.unstack(-1)
        .round(4)
        .loc[:, ['train', 'test']]
    )
    return res


# Pour pouvoir stocker les scores dans MLFlow, on les convertit en dictionnaires

# In[33]:


def scores_to_dict(score_df):
    d = score_df['train'].to_dict()
    d1 = dict(zip([x+'_train_' for x in  list(d.keys())], list(d.values())))
    d = score_df['test'].to_dict()
    d2 = dict(zip([x+'_test' for x in  list(d.keys())], list(d.values())))
    d1.update(d2)
    return d1


# Création d'une fonction affichant une matrice de confusion

# In[34]:


def plot_cm(y_test, y_pred, target_names=[-1, 0, 1], 
            figsize=(5,3)):
    """Create a labelled confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='g', cmap='BuGn', cbar=False, 
                ax=ax)
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted')
    ax.set_xticklabels(target_names)
    ax.set_ylabel('Actual')
    ax.set_yticklabels(target_names, 
                       fontdict={'verticalalignment': 'center'});


# #### Création de la fonction d'entraînement générique

# La fonction suivante est celle qui sera systématiquement appélée pour entraîner les pipelines

# :::{tip}
# L'évaluation fianle des modèles se faisant sur base de f1-macro dans le TD, c'est la métrique que nosu avons retenue pour la partie optimisation de la fonction générique
# :::

# In[43]:


def trainPipelineMlFlow(mlf_XP, 
                        xp_name_iter, 
                        pipeline, 
                        X_train, y_train, X_test, y_test, 
                        target_col='sentiment', 
                        fixed_params={}, 
                        use_opti=False, iterable_params={}, n_iter=20):
    """
    Fonction générique permettant d'entrainer et d'optimiser un pipeline sklearn
    Les paramètres et résultats sont stockés dans MLFlow
    
    INPUTS:
        - mlf_XP : nom de l'experiment à créer dans MLFlow
        - xp_name_iter : nom du run créé dans l'experiment de MLFlow
        - pipeline : un pipeline au sens ed sklearn
        - X_train, y_train, X_test, y_test : des dataframes contenant les jeux d'entrainement et de test
        - target_col : le nom de la colonne du DataFrame y qui constitue la cible
        - fixed_params : un dictionnaire contenant les paramètres fixes dont l'utilisateur souhaite fixer la valeur dans le pipeline
        - use_opti : boolean, est-ce qu'une optimisation est recherchée. Si oui, utilisera RandomizedSearchCV
        - iterable_params : un dictionnaire contenant les nom des paramètres ciblés du pipeline et des listes contenant les valeusr possibles
        - n_iter : le nombre d'itérations maximales à réaliser par RandomizedSearchCV
    
    FONCTIONNEMENT:
        stocke dans MLFlow :
        - le pipeline entrainé
        - les principaux paramètres correspondant aux paramètres fixes et aux éventuels paramètres optimaux après RandomizedSearchCV
        - les scores (scalaires) calculés par la fonction score_estimator
        - le temps d'exécution
        
        imprime :
        - le nom de l'experiment
        - le pipeline entraîné
        - les paramètres principaux (cf FONCTIONNEMENT)
        - la matrice de confusion du pipeline sur le jeu de test fourni en entrée
    
    OUTPUTS:
        - le pipeline entraîné
    
       
    """
  
    mlflow.set_experiment(mlf_XP)

    with mlflow.start_run(run_name=xp_name_iter):
        
        start_time = time.monotonic()  
        
        warnings.filterwarnings("ignore")
        
        # fit pipeline
        pipeline.set_params(**fixed_params)
        if not use_opti:
            search = pipeline
        else:
            search = RandomizedSearchCV(estimator = pipeline, 
                                        param_distributions = iterable_params, 
                                        n_jobs = -1, 
                                        cv = 5, 
                                        scoring = 'f1_macro', 
                                        n_iter = n_iter)
        
        search.fit(X_train, y_train[target_col])
                
        # get params
        params_to_log = fixed_params #select initial params
        if use_opti:
            params_to_log.update(search.best_params_) #update for optimal solution
        mlflow.log_params(params_to_log)
        
        # Evaluate metrics
        y_pred=search.predict(X_test)
        score = score_estimator(estimator=search, 
                                         X_train=X_train, 
                                         X_test=X_test, 
                                         df_train=y_train, 
                                         df_test=y_test, 
                                         target_col=target_col
                                )
        
        # Print out metrics
        print('XP :', xp_name_iter, '\n')
        print('pipeline : \n', search, '\n')
        print("params: \n", params_to_log, '\n')
        print('scores : \n', score, '\n')
        print("Test confusion matrix: \n")
        plot_cm(y_test, search.predict(X_test))
        
        
        #r Report to MlFlow
        mlflow.log_metrics(scores_to_dict(score))
        mlflow.sklearn.log_model(pipeline, xp_name_iter)
        
        end_time = time.monotonic()
        elapsed_time = timedelta(seconds=end_time - start_time)
        print('elapsed time :', elapsed_time)
        mlflow.set_tag(key="elapsed_time", value=elapsed_time)   
        
        
        
    return search;
        


# #### Utilitaires : pour faciliter l'utilisation des pipelines

# Si les pipelines permettent un traitement souple et homogène entre les jeux de données, leur manipulation n'est pas évidente.
# Notamment, le libellé des paramètres peut vide devenir délicat et difficilement lisible avec une combinaison de nom d'étape et du nom du paramètre dans l'étape du pipeline.
# La fonction suivante permet de rechercher tous les paramètres d'un pipeline qui contiennent une chaine de caractère spécifique.

# In[36]:


def target_params(pipe, dict_keyval):
    """
    Crée un dictionnaire constitué de tous les paramètres incluant 'pattern' d'un pipe et leur assigne une valeur unique
    """
    
    res={}
    for key in list(dict_keyval.keys()):
    
        target = "[a-zA-Z\_]+__" + key

        rs = re.findall(target, ' '.join(list(pipe.get_params().keys())))
        rs=dict.fromkeys(rs, dict_keyval[key])
        res.update(rs)
    return res


# #### Utilitaires : Adaptation des pipelines

# La cellule suivante permet de créer des étapes de sélection de colonnes dans les Data Frame en entrée

# In[37]:


from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]

class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]


# ### Approche initiale

# On commence par construire un modèle simple qui nous servira de modèle de base que nous chercherons à améliorer.

# :::{warning}
# 
# Dans cette première étape, nous travaillerons sur le jeu `train` que nous avon découpé et évaluerons ses performances sur le jeu `val`.
# 
# Seuls les principaux modèles seront réentrainés sur `{ train + val }` avant d'être évalués sur le véritable jeu `test`
# 
# :::

# #### Bag of Words avec Random Forest

# Dans cette expérimentation, nous créons un modèle simple :
# 
# ![BoW_RF](images/Pipeline_BoW_RF.png)

# In[38]:


bow_pipeline = Pipeline(
    steps=[
        ('coltext', TextSelector('text')), #Sélection de la colonne à transformer (corpus)
        ("tfidf", TfidfVectorizer()),
        ("classifier", RandomForestClassifier(n_jobs=-1)),
    ]
)


# Déjà dans cet exemple simple, le nombre de paramètres est important et leur nom vite complexe :

# In[39]:


list(bow_pipeline.get_params().keys())


# En première intention on ajuste le pipeline sur le jeu d'entraînement avant les étapes de preprocessing réalisées lors de l'EDA

# In[45]:


base_TfIdf_RF_= trainPipelineMlFlow(
                    mlf_XP = "Rapport",
                    xp_name_iter = "base_TfIdf_RF", 
                    pipeline = bow_pipeline, 
                    X_train = X_train, y_train = y_train, X_test = X_val, y_test = y_val,
                    target_col = 'sentiment',
                    fixed_params = {'classifier__random_state':42}
                    );


# Le modèle de base produit un f1 macro de 67,5% sur le jeu de validation avec le paramétrage par défaut de sklearn.
# On observe le très fort f1 macro sur le jeu d'entraînement qui indique un fort surapprentissage.
# L'intérêt de ce pipeline est d'être très rapide à l'entraînement (à peine plus de 3 secondes ici)

# #### variante preprocessing

# In[48]:


X_train_prepro


# In[51]:


X_val_prepro


# In[52]:


base_TfIdf_RF_prepro_= trainPipelineMlFlow(
                            mlf_XP = "Rapport",
                            xp_name_iter = "base_TfIdf_RF_prepro", 
                            pipeline = bow_pipeline, 
                            X_train = X_train_prepro, y_train = y_train, X_test = X_val_prepro, y_test = y_val,
                            target_col = 'sentiment',
                            fixed_params = {'classifier__random_state':42}
                        );


# :::{admonition} apport du preprocessing
# On observe tout de suite l'apport des retraitemenst effectués à l'étape EDA : le modèle est passé à une performance de **70,47%** sur le jeu de validation sans autres modifications
# :::

# #### variante optimisée

# Une variante consiste à essayer d'ajuster lees hyper paramètres du pipeline dans l'espoire de gagner en performance

# In[41]:


params = {
    "tfidf__use_idf": [True, False],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1,3)],
    "classifier__bootstrap": [True, False],
    "classifier__class_weight": ["balanced", None],
    "classifier__n_estimators": [100, 300, 500, 800, 1200],
    "classifier__max_depth": [5, 8, 15, 25, 30],
    "classifier__min_samples_split": [2, 5, 10, 15, 100],
    "classifier__min_samples_leaf": [1, 2, 5, 10]
}

trainPipelineMlFlow(
                    mlf_XP="DSA_Tweets",
                    xp_name_iter="Bag Of Words - RF-Opti - n_iter_30", 
                    pipeline=bow_pipeline, 
                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    target_col='sentiment',
                    fixed_params={'classifier__random_state':42},
                    use_opti=True,
                    iterable_params=params,
                    n_iter=30
                    )


# ### Bag of Words avec régression logistique

# In[45]:


bow_pipeline_LR = Pipeline(
    steps=[
        ('coltext', TextSelector('text')), #Sélection de la colonne à transformer (corpus)
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(solver='liblinear', multi_class='auto')),
    ]
)


# In[120]:


list(bow_pipeline_LR.get_params().keys())


# In[54]:


params = {
    "tfidf__use_idf": [True, False],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1,3)]
}    

trainPipelineMlFlow(
                    mlf_XP="DSA_Tweets",
                    xp_name_iter="Bag Of Words - LR-Opti - n_iter_30", 
                    pipeline=bow_pipeline_LR, 
                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    target_col='sentiment',
                    fixed_params={'classifier__random_state':42},
                    use_opti=True,
                    iterable_params=params,
                    n_iter=30
                    )


# In[69]:


pipe = bow_pipeline_LR


params = target_params(pipe, {
    "use_idf": [True, False],
    "ngram_range": [(1, 1), (1, 2), (1,3), (1,4)]
})



trainPipelineMlFlow(
                    mlf_XP="DSA_Tweets",
                    xp_name_iter="Bag Of Words - LR-Opti - n_iter_30", 
                    pipeline = pipe, 
                    X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs':-1,'random_state':42}),
                    use_opti = True,
                    iterable_params = params,
                    n_iter = 30
                    )


# In[68]:


pipe = bow_pipeline_LR_prepro


params = target_params(pipe, {
    "use_idf": [True, False],
    "ngram_range": [(1, 1), (1, 2), (1,3), (1,4)]
})



trainPipelineMlFlow(
                    mlf_XP="DSA_Tweets",
                    xp_name_iter="Bag Of Words - LR-Opti - n_iter_30", 
                    pipeline = pipe, 
                    X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs':-1,'random_state':42}),
                    use_opti = True,
                    iterable_params = params,
                    n_iter = 30
                    )


# In[123]:


trainPipelineMlFlow(
                    mlf_XP="DSA_Tweets",
                    xp_name_iter="Bag Of Words - LR", 
                    pipeline=bow_pipeline_LR, 
                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    target_col='sentiment',
                    fixed_params={'classifier__random_state':42}
                    )


# ## TextPreprocessor

# In[78]:


from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 apply_lemmatizer=True,
                 apply_lowercase=True,
                 apply_url_standerdisation=True,
                 apply_user_standerdisation=True,
                 apply_emoticon_to_words=True,
                 apply_stopwords_removal=True,
                 apply_shortwords_removal=True,
                 apply_non_alphabetical_removal=True,
                 apply_only_2_consecutive_charac=True):
        
        self.apply_lemmatizer = apply_lemmatizer
        self.apply_lowercase = apply_lowercase
        self.apply_url_standerdisation = apply_url_standerdisation
        self.apply_user_standerdisation = apply_user_standerdisation
        self.apply_emoticon_to_words = apply_emoticon_to_words
        self.apply_stopwords_removal = apply_stopwords_removal
        self.apply_shortwords_removal = apply_shortwords_removal
        self.apply_non_alphabetical_removal = apply_non_alphabetical_removal
        self.apply_only_2_consecutive_charac = apply_only_2_consecutive_charac
        
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        res= preprocess_text(X, 
                               apply_lemmatizer = self.apply_lemmatizer,
                               apply_lowercase = self.apply_lowercase,
                               apply_url_standerdisation = self.apply_url_standerdisation,
                               apply_user_standerdisation = self.apply_user_standerdisation,
                               apply_emoticon_to_words = self.apply_emoticon_to_words,
                               apply_stopwords_removal = self.apply_stopwords_removal,
                               apply_shortwords_removal = self.apply_shortwords_removal,
                               apply_non_alphabetical_removal = self .apply_non_alphabetical_removal,
                               apply_only_2_consecutive_charac = self.apply_only_2_consecutive_charac
                              )
        return res


# In[79]:


bow_pipeline_LR_prepro = Pipeline(
    steps=[
        ('coltext', TextSelector('text')), #Sélection de la colonne à transformer (corpus)
        ('prepro', TextPreprocessor()), 
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(solver='liblinear', multi_class='auto')),
    ]
)


# In[80]:


list(bow_pipeline_LR_prepro.get_params().keys())


# In[81]:


trainPipelineMlFlow(
                    mlf_XP = "DSA_Tweets",
                    xp_name_iter = "Bag Of Words - LRprepro", 
                    pipeline = bow_pipeline_LR_prepro, 
                    X_train = X_train , y_train = y_train , X_test = X_test , y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs': -1, 'random_state':42})
                    )


# In[35]:


target_params(bow_pipeline_LR_prepro, {'n_jobs': -1, 'random_state':42})


# In[82]:


pipe = bow_pipeline_LR_prepro

trainPipelineMlFlow(
                    mlf_XP = "DSA_Tweets",
                    xp_name_iter = "Bag Of Words - LRprepro", 
                    pipeline = pipe, 
                    X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs': -1, 'random_state':42, 'apply_emoticon_to_words':False})
                    )


# ### Ici

# params = target_params(pipe, {'apply_emoticon_to_words': [True, False]
#                               ,
#                               'apply_lemmatizer': [True, False],
#                               'apply_lowercase': [True, False],
#                               'apply_non_alphabetical_removal': [True, False],
#                               'apply_shortwords_removal': [True, False],
#                               'apply_stopwords_removal': [True, False],
#                               'apply_url_standerdisation': [True, False],
#                               'apply_user_standerdisation': [True, False]
#                               })
# 

# In[85]:


pipe = bow_pipeline_LR


params = target_params(pipe, 
                       {"use_idf": [True, False]}
                      )


pipe = bow_pipeline_LR_prepro

trainPipelineMlFlow(
                    mlf_XP = "DSA_Tweets",
                    xp_name_iter = "Bag Of Words - LRprepro - Opti", 
                    pipeline = pipe, 
                    X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs': -1, 'random_state':42}),
                    use_opti = True,
                    iterable_params = params
                    )


# In[164]:


pipe = bow_pipeline_LR_prepro

params = target_params(pipe, {'apply_emoticon_to_words': [True, False],
                              'apply_lemmatizer': [True, False],
                              'apply_lowercase': [True, False],
                              'apply_non_alphabetical_removal': [True, False],
                              'apply_shortwords_removal': [True, False],
                              'apply_stopwords_removal': [True, False],
                              'apply_url_standerdisation': [True, False],
                              'apply_user_standerdisation': [True, False]
                              })

params


# In[89]:


pipe = bow_pipeline_LR_prepro


params = target_params(pipe, {
    "use_idf": [True, False]
})



trainPipelineMlFlow(
                    mlf_XP="DSA_Tweets",
                    xp_name_iter="Bag Of Words - LR-Opti - n_iter_30", 
                    pipeline = pipe, 
                    X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs':-1,'random_state':42}),
                    use_opti = True,
                    iterable_params = params,
                    n_iter = 30
                    )


# In[102]:


X_train_prepro = pd.DataFrame(preprocess_text(X_train['text']), columns=['text'])


# In[103]:


X_train_prepro


# In[104]:


X_test_prepro = pd.DataFrame(preprocess_text(X_test['text']), columns=['text'])


# In[105]:


X_test


# In[106]:


pipe = bow_pipeline_LR


params = target_params(pipe, {
    "use_idf": [True, False]
})



trainPipelineMlFlow(
                    mlf_XP="DSA_Tweets",
                    xp_name_iter="Bag Of Words - LR-prepro", 
                    pipeline = pipe, 
                    X_train = X_train_prepro, y_train = y_train, X_test = X_test_prepro, y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs':-1,'random_state':42}),
                    use_opti = True,
                    iterable_params = params,
                    n_iter = 30
                    )


# # PyTorch

# In[29]:


import torch
torch.cuda.is_available()


# In[30]:


from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import pipeline


import numpy as np
from scipy.special import softmax
import csv
import urllib.request


# In[31]:



# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []


    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# In[32]:


task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

model = AutoModelForSequenceClassification.from_pretrained('/mnt/pretrained_models/'+MODEL)
tokenizer = AutoTokenizer.from_pretrained('/mnt/pretrained_models/'+MODEL)
config = AutoConfig.from_pretrained('/mnt/pretrained_models/'+MODEL)


# In[33]:


# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]


# In[34]:


nlp=pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)


# In[35]:


def TorchTwitterRoBERTa_Pred(text = "Good night 😊"):
    text = preprocess(text)
    otpt = nlp(text)[0]
#    otpt = (list(otpt[i].values())[1] for i in range(len(otpt)))
    neg = otpt[0]['score']
    neu = otpt[1]['score']
    pos = otpt[2]['score']
    
#    NewName = {0:'roBERTa-neg', 1:'roBERTa-neu', 2:'roBERTa-pos'}
#    otpt = pd.json_normalize(otpt).transpose().rename(columns=NewName).reset_index().drop([0]).drop(columns=['index'])
    return neg, neu, pos


# In[36]:


test = TorchTwitterRoBERTa_Pred()
test


# In[37]:


def run_loopy_roBERTa(df):
    v_neg, v_neu, v_pos = [], [], []
    for _, row in df.iterrows():
        v1, v2, v3 = TorchTwitterRoBERTa_Pred(row.values[0])
        v_neg.append(v1)
        v_neu.append(v2)
        v_pos.append(v3)
    df_result = pd.DataFrame({'roBERTa_neg': v_neg,
                              'roBERTa_neu': v_neu,
                              'roBERTa_pos': v_pos})
    return df_result


# In[38]:


class clTwitterroBERTa(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        res = run_loopy_roBERTa(X[[self.field]])
        
        #self.res[['roBERTa_neg', 'roBERTa_neu', 'roBERTa_pos']] =  X[self.field].apply(lambda x : TorchTwitterRoBERTa_Pred(x)).apply(pd.Series)
        return res
        #return self.res


# In[39]:


roBERTa_pipe=Pipeline([
                     ('roBERTa', clTwitterroBERTa(field='text'))
                    ])


# In[40]:


roBERTa_RF_Pipe = Pipeline(
    steps=[
        ('roBERTa', roBERTa_pipe),
        ("classifier", RandomForestClassifier(n_jobs=-1))
    ]
)


# In[130]:


pipe = roBERTa_RF_Pipe


trainPipelineMlFlow(
                    mlf_XP="DSA_Tweets",
                    xp_name_iter="roBERTa - LR", 
                    pipeline = pipe, 
                    X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs':-1,'random_state':42})
                    )


# ### Transformation des données par roBERTa

# In[133]:


import gc

gc.collect()

torch.cuda.empty_cache()


# In[41]:


import torch
torch.cuda.empty_cache()


# In[42]:


X_train_roBERTa = roBERTa_pipe.transform(X_train)


# In[43]:


X_train_roBERTa


# In[44]:


X_test_roBERTa = roBERTa_pipe.transform(X_test)


# In[72]:


X_train_roBERTa.to_parquet('/mnt/data/interim/X_train_roBERTa.gzip',compression='gzip')
X_test_roBERTa.to_parquet('/mnt/data/interim/X_test_roBERTa.gzip',compression='gzip')


# In[ ]:





# In[45]:


roBERTa_RF = Pipeline(
    steps=[
        ("classifier", RandomForestClassifier(n_jobs=-1))
    ]
)


# In[57]:


pipe = roBERTa_RF

params = target_params(pipe, {
    "bootstrap": [True, False],
    "class_weight": ["balanced", None],
    "n_estimators": [100, 300, 500, 800, 1200],
    "max_depth": [5, 8, 15, 25, 30],
    "min_samples_split": [2, 5, 10, 15, 100],
    "min_samples_leaf": [1, 2, 5, 10]
})


roBERTa_RF_=trainPipelineMlFlow(
                    mlf_XP="DSA_Tweets",
                    xp_name_iter="roBERTa - RF - opti - 30", 
                    pipeline = pipe, 
                    X_train = X_train_roBERTa, y_train = y_train, X_test = X_test_roBERTa, y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs':-1,'random_state':42}),
                    use_opti = True,
                    iterable_params=params,
                    n_iter=30
                    )


# ### roBERTa + xgBoost

# https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

# ### Essai combinaison de différentes méthodes

# In[48]:


class Blob(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X[['polarity', 'subjectivity']] =  X[self.field].apply(lambda x:TextBlob(x).sentiment).apply(pd.Series)
        return X[['polarity', 'subjectivity']]


# In[49]:


blob_pipe=Pipeline([
                     ('blob', Blob(field='text'))
                    ])


# In[50]:


X_train_Blob=blob_pipe.transform(X_train)
X_train_Blob.head()


# In[51]:


X_test_Blob=blob_pipe.transform(X_test)
X_test_Blob.head()


# In[73]:


X_train_Blob.to_parquet('/mnt/data/interim/X_train_Blob.gzip',compression='gzip')
X_test_Blob.to_parquet('/mnt/data/interim/X_test_Blob.gzip',compression='gzip')


# In[52]:


class Vader(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
        sid = SentimentIntensityAnalyzer()
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        sid = SentimentIntensityAnalyzer()
        X[['neg', 'neu', 'pos', 'compound']] =  X[self.field].apply(sid.polarity_scores).apply(pd.Series)
        return X[['neg', 'neu', 'pos', 'compound']]


# In[53]:


vader_pipe=Pipeline([
                     ('vader', Vader(field='text'))
                    ])


# In[54]:


X_train_Vader=vader_pipe.transform(X_train)
X_train_Vader.head()


# In[55]:


X_test_Vader=vader_pipe.transform(X_test)
X_test_Vader.head()


# In[74]:


X_train_Vader.to_parquet('/mnt/data/interim/X_train_Vader.gzip',compression='gzip')
X_test_Vader.to_parquet('/mnt/data/interim/X_test_Vader.gzip',compression='gzip')


# In[56]:


X_train_compound = pd.concat([X_train_roBERTa, X_train_Blob, X_train_Vader], axis=1)
X_test_compound = pd.concat([X_test_roBERTa, X_test_Blob, X_test_Vader], axis=1)


# In[57]:


X_train_compound.head()


# In[58]:


X_test_compound.head()


# In[105]:


pipe = roBERTa_RF

params = target_params(pipe, {
    "bootstrap": [True, False],
    "class_weight": ["balanced", None],
    "n_estimators": [100, 300, 500, 800, 1200],
    "max_depth": [5, 8, 15, 25, 30],
    "min_samples_split": [2, 5, 10, 15, 100],
    "min_samples_leaf": [1, 2, 5, 10]
})


roBERTa_RF_=trainPipelineMlFlow(
                    mlf_XP="DSA_Tweets",
                    xp_name_iter="roBERTa_Blob_Vader - RF - opti - 30", 
                    pipeline = pipe, 
                    X_train = X_train_compound, y_train = y_train, X_test = X_test_compound, y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs':-1,'random_state':42}),
                    use_opti = True,
                    iterable_params=params,
                    n_iter=30
                    )


# In[70]:


import xgboost as xgb


# In[69]:


roBERTa_xgb = Pipeline(
    steps=[
        ("classifier", xgb.XGBClassifier())
    ]
)


# In[60]:


pipe = roBERTa_xgb

params = target_params(pipe, {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     })


roBERTa_xgb_ = trainPipelineMlFlow(
                    mlf_XP="DSA_Tweets",
                    xp_name_iter="roBERTa - xgb - opti", 
                    pipeline = pipe, 
                    X_train = X_train_compound, y_train = y_train, X_test = X_test_compound, y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs':-1,'random_state':42}),
                    use_opti = True,
                    iterable_params=params,
                    n_iter=20
                    )


# In[ ]:





# ### Essai opti F1

# In[31]:


pipe = bow_pipeline


essai_=trainPipelineMlFlow(
                    mlf_XP="opti F1",
                    xp_name_iter="test", 
                    pipeline = pipe, 
                    X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                    target_col = 'sentiment',
                    fixed_params = target_params(pipe, {'n_jobs':-1,'random_state':42}),
                    use_opti = False
                    )


# In[32]:


essai_.predict_proba(X_train)


# In[33]:


X_train.head()


# In[52]:


for var in [-1, 0, 1]:
    plt.figure(figsize=(12,4))
    sns.distplot(essai_.predict_proba(X_train)[(y_train['sentiment']==var),0], bins=30, kde=False, 
                 color='green', label='Negative')
    sns.distplot(essai_.predict_proba(X_train)[(y_train['sentiment']==var),1], bins=30, kde=False, 
                 color='red', label='Neutral')
    sns.distplot(essai_.predict_proba(X_train)[(y_train['sentiment']==var),2], bins=30, kde=False, 
                 color='blue', label='Positive')
    plt.legend()
    plt.title(f'Histogram of {var} by true sentiment');


# Stratégie : on maximise le seuil pour la décision positive, puis sur les non positifs, on maximise le seuil pour les négatifs, le reste est neutre

# In[65]:


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


# In[62]:


def find_optimal_f1_thresholds(pipe, X, y):
    
    probs = pipe.predict_proba(X)
    
    # keep probabilities for the positive outcome only
    pos_probs = probs[:,2]
    # define thresholds
    thresholds = np.arange(0, 1, 0.001)
    # evaluate each threshold
    scores = [f1_score([(1 if i==1 else 0) for i in y ], to_labels(pos_probs, t)) for t in thresholds]
    # get best threshold
    ix = np.argmax(scores)

    
    res = {'pos_threshold' : thresholds[ix], 'pos_f1' : scores[ix] }
    
    # keep probabilities for the positive outcome only
    neg_probs = probs[:,0]
    # define thresholds
    thresholds = np.arange(0, 1, 0.001)
    # evaluate each threshold
    scores = [f1_score([(1 if i==-1 else 0) for i in y ], to_labels(neg_probs, t)) for t in thresholds]
    # get best threshold
    ix = np.argmax(scores)

    
    res.update({'neg_threshold' : thresholds[ix], 'neg_f1' : scores[ix] })
    
    return res
    


# In[119]:


thres = find_optimal_f1_thresholds(roBERTa_RF_, X_train_compound, y_train['sentiment'])


# In[120]:


thres


# In[109]:


y_train['sentiment']


# In[110]:


roBERTa_RF_.predict_proba(X_train_compound)


# In[63]:


def sentiment_predict(pipe, X, dict_thres):
    seuil_pos=dict_thres['pos_threshold']
    seuil_neg=dict_thres['neg_threshold']

    probs = pipe.predict_proba(X)

    y_test_pred_pos = to_labels(probs[:,2], seuil_pos)
    y_test_pred_neg = to_labels(probs[:,0], seuil_neg)

    y_test_pred = y_test_pred_pos
    y_test_pred[(y_test_pred_pos==0)] = -y_test_pred_neg[(y_test_pred_pos==0)]
    return y_test_pred


# In[122]:


y_test_pred = sentiment_predict(roBERTa_RF_, X_test_compound,thres)


# In[123]:


f1_score(y_test, y_test_pred, average='macro')


# In[66]:


thres_xgb = find_optimal_f1_thresholds(roBERTa_xgb_, X_train_compound, y_train['sentiment'])


# In[67]:


y_test_pred_xgb = sentiment_predict(roBERTa_xgb_, X_test_compound,thres_xgb)


# In[68]:


f1_score(y_test, y_test_pred_xgb, average='macro')


# # SHAP

# In[76]:


import shap

shap.initjs()


# # sujets

# In[ ]:


import gensim.corpora as corpora# Create Dictionary
id2word = corpora.Dictionary(data_words)

