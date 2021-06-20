#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
# **TD DSA 2021 de Antoine Ly   -   rapport de Fabien Faivre**
# -------------------------     -------------------------------------

# # Chargement initial des données

# Dans cette section nous chargeons et séparons les données

# # Setup

# In[25]:


#Temps et fichiers
import os
import warnings
import time
from datetime import timedelta

#Manipulation de données
import pandas as pd
import numpy as np

#Tracking d'expérience
import mlflow
import mlflow.sklearn


# In[82]:


#Cellule strictement technique qui permet de sauver les exigences pour recréer au besoin l'image docker du projet
get_ipython().system('pip freeze > /mnt/docker/requirements.txt')


# ## Utilisation du package

# Durent ce projet, certaines parties du code ont été re packagées dans un package propre au projet afin de factliter la lecture du core et permettre la réutilisabilité des développements

# In[6]:


#Cette cellule permet d'appeler la version packagée du projet et d'en assurer le reload avant appel des fonctions
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[7]:


from dsa_sentiment.scripts.make_dataset import load_data
from dsa_sentiment.scripts.evaluate import eval_metrics
from dsa_sentiment.scripts.make_dataset import Preprocess_StrLower, Preprocess_transform_target


# ## Configuration de l'experiment MLFlow

# [MLFlow](https://mlflow.org/) sera utilisé comme outil de suivi et de stockage des expérimentatiosn réalisées

# In[8]:


mlflow.tracking.get_tracking_uri()


# # Chargement des données

# In[9]:


get_ipython().system('pwd')


# In[10]:


data_folder = os.path.join('/mnt', 'data', 'raw')
all_raw_files = [os.path.join(data_folder, fname)
                    for fname in os.listdir(data_folder)]
all_raw_files


# In[11]:


random_state=42


# Il n'est pas possible de faire de l'imputation comme avec des champs numérique. Il convient donc de supprimer les tweets vides (`dropNA=True`).

# On laisse 20% de données de côté dans un jeu de validation. Afin de simuler des conditions réelles d'expoitation, le classement des modèles se fera sur le jeu de validation uniquement sans toucher au jeu de test.
# 
# A l'issue du premier classement les modèles seront réentrainés sur `train + validation` avant d'être évalués sur le jeu de test

# In[12]:


X_train, y_train, X_val, y_val = load_data(all_raw_files[2], split=True, test_size=0.2, random_state=random_state, dropNA=True)


# In[13]:


X_train.head()


# In[14]:


print(f'le jeu d\'entraînement initial contient', X_train.shape[0] + X_val.shape[0] , 'lignes')
print(f'le jeu d\'entraînement retenu contient', X_train.shape[0] , 'lignes')
print(f'le jeu de validation retenu contient', X_val.shape[0] , 'lignes')


# In[15]:


y_train.head()


# In[16]:


X_test, y_test = load_data(all_raw_files[1], split=False, random_state=random_state, dropNA=True)


# In[17]:


X_test.head()


# In[18]:


print(f'le jeu de test contient', X_test.shape[0] , 'lignes')


# ## Transformation initiales des données

# Cette partie vise uniquement à sélectionner les colonnes dont nous nous servirons et à transcoder la cible au format souhaité.

# In[19]:


# Dans ce projet on ne se servira que du champs `text`. On cherche toutefois à conserver le format pandas DataFrame
X_train = X_train[['text']]
X_val = X_val[['text']]
X_test = X_test[['text']]


# In[20]:


X_train.head()


# ## Préalable : transformation des sorties

# On commence par transformer les cibles pour se conformer aux instructions

# In[21]:


y_train = Preprocess_transform_target(y_train, columns_to_process=['sentiment'])
y_train.head()


# In[22]:


y_val = Preprocess_transform_target(y_val, ['sentiment'])
y_val.head()


# In[23]:


y_test = Preprocess_transform_target(y_test, ['sentiment'])
y_test.head()


# ## On exporte les données sous parquet pour avoir une source de vérité unique dans les notebooks

# In[24]:


# Données explicatives
X_train.to_parquet('/mnt/data/interim/X_train.gzip',compression='gzip')
X_val.to_parquet('/mnt/data/interim/X_val.gzip',compression='gzip')
X_test.to_parquet('/mnt/data/interim/X_test.gzip',compression='gzip')

# Données à expliquer
y_train.to_parquet('/mnt/data/interim/y_train.gzip',compression='gzip')
y_val.to_parquet('/mnt/data/interim/y_val.gzip',compression='gzip')
y_test.to_parquet('/mnt/data/interim/y_test.gzip',compression='gzip')

