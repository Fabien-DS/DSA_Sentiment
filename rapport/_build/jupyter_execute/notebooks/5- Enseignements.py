#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
# **TD DSA 2021 de Antoine Ly   -   rapport de Fabien Faivre**
# -------------------------     -------------------------------------

# # Enseignements et pistes d'amélioration

# Plusieurs modèle ont été testés. Le modèle champion est un XGBoost optimisé s'appuyant sur une combinaison de features créées à partir de modèles pré entrainé dont roBERTa tweet.
# Ce modèle permet d'atteindre un f1 macro de **76%** sur le jeu test

# In[3]:


import pandas as pd


# In[4]:


res_fin3 = pd.read_parquet('/mnt/data/processed/res_fin3.gzip')
res_fin3


# Ce projet a été une constant source d'étonnement.
# 
# Le fait de disposer de 3 classes à prédire a été un élément complexifiant par rapport au cas binaire. L'absence de courbe ROC nous rend beaucoup plus dépendant des chiffres.
# 
# Après la découverte de [twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment), je pensais que le sujet serait plié. Après tout ce modèle a été entrainé explicitement pour ce cas (58 millions de tweets, en anglais et optimisé pour l'analyse de sentiments). Je pensais voir le f1 macro s'envoler, ce qui n'a pas été le cas. Le modèle a bien aidé, mais le gain est resté modeste (6 points de f1 macro par rapport aux approches fréquentistes classiques).
# 
# Au final en analysant les fausses prédictions, on réalise que la labelisation de plusieurs tweets laisse songeur.
# Ceci met en lumière le fait que l'appréciation de la tonalité n'est pas toujours évidente et que des erreurs humaines peuvent en plus se glisser.
# Si ce phénomène existe déjà pour les catégories extrèmes (`positif` et `négatif`) on imagine la sensibilité pour la classe générique `neutre`...
# 
# Par ailleurs rien n'indique que la stratégie de labellisation utilisé dans ce cas corresponde à celle utilisée pour le pré entrainement de roBERTa tweet.
# 
# Après un acquis de conscience, un entrainement réel d'un modèle de deep learning s'appuyant sur les modèles préentrainées `BERT`, `DistilBERT` et surtout `RoBERTa` en en réentrainant la dernière couche pour le sujet étudié a de loin présenté le meilleur gain (+3,4% de f1_macro par rapport au modèle RoBERTa utilisé directement en entrée d'un modèle classique). Ce gain est vraisemblablement à mettre au crédit de la dimention plus élevée de l'avant dernière couche et à la labellisation spéciale du projet.
# 
# Deux pistes auraient pu être explorées pour améliorer la performance :
# - potentiellement modéliser le sujet discuté dans les tweets et le rajouter comme feature. On avait en effet vu que les tweets positifs par exemple se rapportaient principalement à la fête des mère et au `star wars day`
# - effectivement utiliser un modèle ensembliste y compris avec `RoBERTa` rendu impossible ici du fait du temps d'exécution et de la limiattion du matériel utilisé.
# 
# Enfin ce sujet a été l'occasion de se frotter à plusieurs difficultés techniques liées principalement :
# - à l'utilisation de ressources GPU depuis docker
# - à l'utilisation des GPU pour XGBoost (non pris en compte par défaut)
# - aux pipelines sklearn, pratiques mais pas toujours compatibles avec les packages (ex SHAP) et nécessitant souvent des créations de classes ad-hoc
# - une première confrontation réelle avec les modèles de Deep Learning.

# In[ ]:




