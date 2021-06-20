#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------------
# **TD DSA 2021 de Antoine Ly   -   rapport de Fabien Faivre**
# -------------------------     -------------------------------------

# # Analyse descriptive

# In[5]:


get_ipython().system('pip install textblob')


# In[6]:


get_ipython().system('pip install emot')


# In[7]:


get_ipython().system('pip install wordcloud')


# In[8]:


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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# ## Utilisation du package

# In[9]:


#Cette cellule permet d'appeler la version packagée du projet et d'en assurer le reload avant appel des fonctions
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[10]:


from dsa_sentiment.scripts.make_dataset import load_data
from dsa_sentiment.scripts.evaluate import eval_metrics
from dsa_sentiment.scripts.make_dataset import Preprocess_StrLower, Preprocess_transform_target


# ## Configuration de l'experiment MLFlow

# In[11]:


mlflow.tracking.get_tracking_uri()


# ## Chargement des données

# In[12]:


# On Importe les données

#df
df_train=pd.read_parquet('/mnt/data/interim/df_train.gzip')
df_val=pd.read_parquet('/mnt/data/interim/df_val.gzip')
df_test=pd.read_parquet('/mnt/data/interim/df_test.gzip')

#X
X_train=pd.read_parquet('/mnt/data/interim/X_train.gzip')
X_val=pd.read_parquet('/mnt/data/interim/X_val.gzip')
X_test=pd.read_parquet('/mnt/data/interim/X_test.gzip')

#y
y_train=pd.read_parquet('/mnt/data/interim/y_train.gzip')
y_val=pd.read_parquet('/mnt/data/interim/y_val.gzip')
y_test=pd.read_parquet('/mnt/data/interim/y_test.gzip')


# # EDA

# On commence par nalyser l'équilibre des différentes classes de sentiments

# In[13]:


df = df_train
df.head()


# ## Analyse de l'équilibre du jeu d'entrainement par label

# In[14]:


fig = px.histogram(df, x="sentiment", color="sentiment", title = 'Nombre de tweets par sentiment')
fig.show()


# Il existe un léger déséquilibre dans les classes en faveur des sentiments `neutral`

# ## Analyse des champs lexicaux par label

# Pour la suite des travaux, on créée un corpus contenant la concaténation de tous les tweets d'une certaine tonalité.

# In[15]:


def create_corpus(text_series):
    text = text_series.apply(lambda x : x.split())
    text = sum(text, [])
    return text
    


# In[16]:


positive_text = create_corpus(df['text'][df['sentiment']=='positive'])
negative_text = create_corpus(df['text'][df['sentiment']=='negative'])
neutral_text = create_corpus(df['text'][df['sentiment']=='neutral'])


# Il devient alors possible de crééer des histogrammes représentant la fréquence de N-grams dans un corpus =donné

# In[17]:


def plot_freq_dist(text_corpus, nb=30, ngram=1, title=''):
    '''
    Plot the most common words
    
    inputs:
        text_corpus : a corpus of words
        nb : number of words to plot
        title : graph title
    
    returns:
        nothing, plots the graph
    
    '''

    freq_pos=Counter(ngrams(create_corpus(pd.Series(text_corpus)),ngram))
    pos_df = pd.DataFrame({
        "words":[' '.join(items) for items in list(freq_pos.keys())],
        "Count":list(freq_pos.values())
    })
    common_pos= pos_df.nlargest(columns="Count", n=30)

    fig = px.bar(common_pos, x="words", y="Count", labels={"words": "Words", "Count":"Frequency"}, title=title)
    fig.show();


# In[18]:


plot_freq_dist(positive_text, title = 'Most common words associated with positive tweets')


# Le résultat montre la prépondérance des `stopwords`, ces mots d'articulation, qui sont très communs et gènent l'identifiaction de mots clefs propres à un document / ensemble de documents spécifiques.
# 
# Il convient donc d'effectuer des opérations de retraitement du texte pour analyse. 

# ## Preprocessing

# Parmi les éléments propres aux tweets qui peuvent avoir un impact sur la suite on compte :
# 
#  - les mots clefs marqués par un `#`
#  - les noms d'utilisateurs commençant par un `@`
#  - les emoticons et emojis
#  - les nombre de mots en MAJUSCULES
#  - la répétition de caractères pour marquer l'emphase `!!!!`, `looooong`, ou l'autocensure `f***`
#  - les fautes de frappes (mots de moins de 2 caractères)

# Afin de disposer de traitements homogènes, repoductibles et paramétrables, une fonction spécifique est créée. Les différenst paramètres pourront être testés dans les phase de modélistaion ultérieures.

# source [preprocess](https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners)

# In[57]:


def preprocess_text(text_series, 
                    apply_lemmatizer=True,
                    apply_lowercase=True,
                    apply_url_standerdisation=True,
                    apply_user_standerdisation=True,
                    apply_emoticon_to_words=True,
                    apply_stopwords_removal=True,
                    apply_shortwords_removal=True,
                    apply_non_alphabetical_removal=True,
                    apply_only_2_consecutive_charac=True
                   
                   ):
    '''
    Main preprocess function
    
    inputs:
        text_series : a pandas Series object with text to preprocess
    
    outputs:
        a preprocessed pandas Series object
    '''
    
    processedText = []
    
    if apply_lemmatizer:
        # Create Lemmatizer and Stemmer.
        wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = r"[^(\w|\*|(!){2}|#)]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in text_series:
        
        if apply_lowercase:
            tweet = tweet.lower()
        
        if apply_url_standerdisation:
            # Replace all URls with 'URL'
            tweet = re.sub(urlPattern,' URL',tweet)
        
        if apply_user_standerdisation:
            # Replace @USERNAME to 'USER'.
            tweet = re.sub(userPattern,' USER', tweet)  
        
        if apply_emoticon_to_words:
            # Replace all emojis.
            for emo in EMOTICONS:
                #refactor outputs so that we come up with a single word when/if text spliting afterwards
                val = "_".join(EMOTICONS[emo].replace(",","").split())
                val='EMO_'+val
                tweet = tweet.replace(emo, ' '+val+' ')

            for emot in UNICODE_EMO:
                val = "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split())
                val='EMO_'+val
                tweet = tweet.replace(emo, ' '+val+' ')
      
        if apply_only_2_consecutive_charac:
            # Replace 3 or more consecutive letters by 2 letter.
            tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        if apply_non_alphabetical_removal:
            # Replace all non alphabets.
            tweet = re.sub(alphaPattern, " ", tweet)
        

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            if apply_stopwords_removal: 
                if word in stopwords.words('english'):
                    word=''
            else:
                word=word
            #if word not in stopwordlist:
            if apply_shortwords_removal:
                if len(word)<=1:
                    word=''
            else:
                word=word
            # Lemmatizing the word.
            if apply_lemmatizer:
                word = wordLemm.lemmatize(word)
            else:
                word=word
            
            tweetwords += (word+' ')

        processedText.append(tweetwords)
        
    return processedText


# In[20]:


positive_text_prepro = preprocess_text(df['text'][df['sentiment']=='positive'], apply_lemmatizer=False, apply_non_alphabetical_removal=True)


# In[56]:


pd.Series(positive_text_prepro).head()


# In[21]:


neutral_text_prepro = preprocess_text(df['text'][df['sentiment']=='neutral'], apply_lemmatizer=False, apply_non_alphabetical_removal=True)


# In[58]:


pd.Series(neutral_text_prepro).head()


# In[22]:


negative_text_prepro = preprocess_text(df['text'][df['sentiment']=='negative'], apply_lemmatizer=False, apply_non_alphabetical_removal=True)


# In[59]:


pd.Series(negative_text_prepro).head()


# ## Analyses des mots clefs des tweets positifs

# La fonction suivante permettra de réaliser des nuages de mots à partir d'un corpus

# In[23]:


def plotWc(text, stopwords=None, title=''):
    wc = WordCloud(
            stopwords=stopwords,
            width=800,
            height=400,
            max_words=1000,
            random_state=44,
            background_color="white",
            collocations=False
    ).generate(text)
    
    plt.figure(figsize = (10,10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


# In[24]:


plotWc(" ".join(positive_text_prepro), stopwords=stopwords.words('english'), title = "Wordcloud des tweets positifs")


# Les tweets positpositive_text_prepro marqués par la forte reprétsentation de mots à connotation positive `love`, `good`, `happy`.
# 
# Cest a priori graphique peut être confirmé par un graphique de fréquence des mots individuels les plus présents

# In[26]:


plot_freq_dist(create_corpus(pd.Series(positive_text_prepro)), title = 'Most common words associated with positive tweets')


# In[27]:


plot_freq_dist(create_corpus(pd.Series(positive_text_prepro)), ngram=2, title = 'Most common 2grams associated with positive tweets')


# In[28]:


plot_freq_dist(create_corpus(pd.Series(positive_text_prepro)), ngram=3, title = 'Most common 3grams associated with positive tweets')


# In[29]:


plot_freq_dist(create_corpus(pd.Series(positive_text_prepro)), ngram=4, title = 'Most common 4grams associated with positive tweets')


# [**insight**] : Une grande majorité de tweets positifs se rapportent soit à la fête des mère, soit au 4 Mai du fait du jeu de mot avec Star Wars...
# 
# <div>
# <img src=https://upload.wikimedia.org/wikipedia/fr/c/ca/LogoSW4th.png width="400"/>
# </div>
# 
# 
# Cette spécificité sera surement exploitée par les modèles comme un marqueur probable de tweets positifs.

# ## Analyse des mots clefs des tweets neutres

# In[30]:


plotWc(" ".join(pd.Series(neutral_text_prepro)), stopwords=stopwords.words('english'), title = "Wordcloud des tweets neutres")


# In[31]:


plot_freq_dist(create_corpus(pd.Series(neutral_text_prepro)), title = 'Most common words associated with neutral tweets')


# **[Insight]** On peut déjà remarquer que le mot day, qui est le plus fréquent des mots clefs des tweets positifs apparaît aussi en 6ème position des mots neutres.

# In[32]:


plot_freq_dist(create_corpus(pd.Series(neutral_text_prepro)), ngram=2, title = 'Most common 2grams associated with neutral tweets')


# In[33]:


plot_freq_dist(create_corpus(pd.Series(neutral_text_prepro)), ngram=3, title = 'Most common 3grams associated with neutral tweets')


# In[34]:


plot_freq_dist(create_corpus(pd.Series(neutral_text_prepro)), ngram=4, title = 'Most common 4grams associated with neutral tweets')


# [**insight**] : On voit une source de confusion arriver avec les twwets neutres dans la mesure où une proportion significative de ceux-ci se rapportent aussi à la fête des mères et star wars. 

# ## Analyse des mots clefs des tweets négatifs

# In[35]:


plotWc(" ".join(pd.Series(negative_text_prepro)), stopwords=stopwords.words('english'), title = "Wordcloud des tweets négatifs")


# In[36]:


plot_freq_dist(create_corpus(pd.Series(negative_text_prepro)), title = 'Most common words associated with negative tweets')


# In[37]:


plot_freq_dist(create_corpus(pd.Series(negative_text_prepro)), ngram=2, title = 'Most common 2grams associated with negative tweets')


# In[38]:


plot_freq_dist(create_corpus(pd.Series(negative_text_prepro)), ngram=3, title = 'Most common 3grams associated with negative tweets')


# In[39]:


plot_freq_dist(create_corpus(pd.Series(negative_text_prepro)), ngram=4, title = 'Most common 4grams associated with negative tweets')


# [**insight**] : on observe l'utilisation de mots autocensurés (`**`) et de mots très chargés (`hate`)
# Il ne servira à rien de tester des n-gram de dimension 4 ou plus : le nombre d'occurences est trop faible

# ## Vérification : validation de l'occurence de certains patterns dans le texte

# In[41]:


def list_words_with(text_series, search='', nb=30):
    '''
    Cette fonction permet de lister les mots dans un string qui contiennent une certaine chaîne de caractères
    
    inputs :
        - text_series : un pd.Series contennat les chaînes de caractères
        - search : la séquence à rechercher
        - nb : ressortir les nb occurences les plus fréquentes
    
    output :
        - une liste de tuples contenant 
            + le mot contenant la séquence recherchée
            + le nombre d'occurence dans text_series
    
    '''
    
    
    #searchPattern   = f"\w*{search}\w*"
    searchPattern   = f"\w*{search}\w* "
    
    cnt = Counter()
    
    for tweet in text_series:
        # Replace all URls with 'URL'
        tweet = re.findall(searchPattern,tweet)
        for word in tweet:
            cnt[word] += 1
    return cnt.most_common(nb)
    


# In[43]:


#liste des mots incluant auto-censure **
list_words_with(negative_text_prepro, search='\*{2}')


# In[44]:


#nombre d'utilisateurs
list_words_with(negative_text_prepro, search='USER')


# In[45]:


#nombre d'URLs
list_words_with(negative_text_prepro, search='URL')


# In[46]:


#liste des émojis
list_words_with(negative_text_prepro, search='EMO\w+')


# In[47]:


#les mots qui incluents !!
list_words_with(negative_text_prepro, search='!!')


# In[48]:


#les tweets complets qui incluent 'bs' (apparaît dans les 4grams)
list_words_with(negative_text_prepro, search='[\w ]* bs [\w ]*')


# In[49]:


#listing des mots clefs
list_words_with(negative_text_prepro, search='#[(\w*|\d*)]+')


# In[50]:


def user_names(text_list):
    cnt = Counter()
    for text in text_list:
        for word in text.split():
            if word.startswith('@'):
                cnt[word] += 1
    return cnt
    


# In[51]:


user_names(positive_text)


# In[52]:


user_names(positive_text_prepro)


# In[53]:


user_names(negative_text_prepro)


# In[54]:


user_names(neutral_text_prepro)


# # Sortie des données préprocessées pour l'analyse

# In[64]:


X_train_prepro=pd.DataFrame(columns=['text'])
X_val_prepro=pd.DataFrame(columns=['text'])
X_test_prepro=pd.DataFrame(columns=['text'])

X_train_prepro['text'] = preprocess_text(X_train['text'], apply_lemmatizer=False, apply_non_alphabetical_removal=True)
X_val_prepro['text'] = preprocess_text(X_val, apply_lemmatizer=False, apply_non_alphabetical_removal=True)
X_test_prepro['text'] = preprocess_text(X_test, apply_lemmatizer=False, apply_non_alphabetical_removal=True)


# In[65]:


X_train_prepro


# In[66]:


# Données explicatives
X_train_prepro.to_parquet('/mnt/data/interim/X_train_prepro.gzip',compression='gzip')
X_val_prepro.to_parquet('/mnt/data/interim/X_val_prepro.gzip',compression='gzip')
X_test_prepro.to_parquet('/mnt/data/interim/X_test_prepro.gzip',compression='gzip')

