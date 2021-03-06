# Consignes

L'objectif est de vous faire approfondir une notion de machine learning aux travers d'une compétition Kaggle. Kaggle est aujourd'hui un site de compétition incontournable dans le monde du machine learning. Même si la plateforme n'est pas représentative des enjeux opérationnels, elle est néanmoins représentative des sujets d'attention de la communauté scientifique et demeure un bon outil d'apprentissage est de partage.


Le projet consiste donc à utiliser le challenge [Tweet sentiment extraction](https://www.kaggle.com/c/tweet-sentiment-extraction/overview/description) à des fins académiques.


## Les données

Les données sont celles proposées par le challenge. Elles composent de deux fichiers:

* `train.csv` ce document comportent les données à utiliser pour calibrer votre modèle. Il comporte toutes les colonnes
* `test.csv` ce document n'est utilisé **que** pour évaluer la performance finale de votre modèle. En aucun cas il ne peut être utilisé pour fine-tuner ou calibrer votre modèle. Il simule les données qui ne sont normalement JAMAIS accessible sur Kaggle (ni dans la vraie vie). à considérer comme un nouvel échantillon.

## Le challenge

Prédire la colonne `sentiment` à partir de la colonne `text`.

## La métrique d'évaluation

On utilisera un score F1 à l'aide de la fonction implémentée dans `scikit-learn`

    from sklearn.metrics import f1_score
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    f1_score(y_true, y_pred, average='macro')


## Labels à utiliser pour la colonne `sentiment`

Vous devrez retraiter la colonne `sentiment` en utilisant les remplacements suivants:

    "neutral"  ->  0
    "negative" -> -1
    "positive" ->  1

## Informations pratiques sur le rendu et la notation.

L'objectif est de se familiariser avec les techniques de text-mining à des fins de classification de sentiments d'un texte. La notation se décomposera en deux parties:

### Notation

* Votre méthodologie et votre approche (12 points) : cette partie doit mettre en avant la motivation des différents retraitements que vous avez appliquez, votre effort de comprendre les implémentations des pacakges que vous aurez utilisés ainsi que le bon sens que leurs utilisations transcrit.
* La performance finale et méthodologie (4 + 4 = 8 points) : cette notation sera relative au groupe. 3 tentatives d'algorithmes/preprocessing différents permettrons de garantir 4 points sur les 8. Les 3 premiers du classement (du groupe 2020) calculé à l'aide du score F1 sur la base de test atteigneront 4 points supplémentaires. Le reste du barême relatif au classement sera dégressif de façon linéaire par palier: les derniers obtenant 1 point minimum.

### Rendu

Le rendu se fera sous la forme d'un court rapport (max 5 pages). Ce dernier peut se faire sous la forme d'un notebook (html ou pdf) ou d'un rapport traditionnel (word, pdf). Il doit mettre en avant la méthodologie employée, les difficultés rencontrées ainsi que les différents apprentissages.


Le projet sera à rendre lors de la séance de **Juillet 2021** de restitution.

## Language de programmation

Il est fortement recommandé d'effectuer le projet en python, mais ceci n'est pas obligatoire.

Bibliographie:

https://www.scor.com/fr/articles-experts/accroitre-vitesse-et-precision-grace-lexploration-de-texte-et-au-traitement
