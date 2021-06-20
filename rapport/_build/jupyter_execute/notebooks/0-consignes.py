#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
# **TD DSA 2021 de Antoine Ly   -   rapport de Fabien Faivre**
# -------------------------     -------------------------------------

# # Description du projet et de la stratégie a priori

# L'approche déployée consiste à analyser des tweets en langue anglaise et de prédire les sentiments qu'ils portent : `{negative: -1, neutral: 0, positive: 1}`

# Dans cet exercice, la langue anglaise est un facteur facilitant dans la mesure où beaucoup de modèles préentrainés existent dans cette langue.

# La difficulté dans cet exercice provient de sa source : les tweets.
# Les approches classiques reposent sur :
# - le passage en minuscule, or dans les tweets, l'utilisation de mots en **majuscules** est un marqueur d'une **émotion forte**
# - l'utilisation de structures linguisitiques relativement correctes augmentées par la lemmatisation / tokenisation. 
# Or, les mots utilisés dans les tweets font l'objet de nombreuses **fautes d'orthographes ou d'abbréviations** (ex `thx`)
# - les marqueurs de ponctuation sont usuellement retirés, or ici, ils peuvent être utilisés comme **smiley** `;-)` ou pour marquer une **émotion forte** `!!!`
# - l'**humour** et les **euphémismes** sont très présents sur tweeter, or les modèles ont beaucoup de mal à distinguer ces cas qui nécessitent une compréhension contextuelle.

# En complément au sujet du TD lui-même, celui-ci a été l'occasion de monter en compétence avec les (je l'espère) bonnes pratiques de codage et l'utilisation de techniques de MLOps.
# 
# Le code de ce projet a été organisé en s'appuyant sur le framework open source [**orbyter**](https://github.com/manifoldai/orbyter-cookiecutter) de la société [Manifold.ai](https://www.manifold.ai/project-orbyter). Ce framework pousse à la standardisation de la structure du code, via l'utilisation de `cookiecutter` et promeut un développement dans un environnement dockerisé dès le départ : 
# 
# ![structure](https://www.manifold.ai/hubfs/Torus.png) 
# 
# La logique de développement pronée est disponible [ici](https://cdn2.hubspot.net/hubfs/4584542/Conference%20Slides/2019StrataNY_EfficientMLengineering.pdf)
# 
# Plusieurs modifications ont dû être apportées aux paramètres du `docker-compose` pour permettre un accès aux ressources GPU depuis le docker.
# 
# Le code a été versionné et est disponible ici [github](https://github.com/Fabien-DS/DSA_Sentiment)
