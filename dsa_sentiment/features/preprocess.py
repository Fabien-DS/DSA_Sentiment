import re
from collections import Counter
from emot.emo_unicode import UNICODE_EMO, EMOTICONS



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
    
    
    searchPattern   = f"\w*{search}\w*"
    
    cnt = Counter()
    
    for tweet in text_series:
        # Replace all URls with 'URL'
        tweet = re.findall(searchPattern,tweet)
        for word in tweet:
            cnt[word] += 1
    return cnt.most_common(nb)


def user_names(text_list):
    cnt = Counter()
    for text in text_list:
        for word in text.split():
            if word.startswith('@'):
                cnt[word] += 1
    return cnt
    
def count_hashtags(df, text_field):
    '''
    count the number of keywords marked by a '#'
    
    inputs : 
        df : a dataframe
        text_field : the name of the text column to analyse
    
    returns :
        a copy of the dataframe df augmented by an additional column 'hashtags_count'
    
    '''
    df['hashtags_count'] = df[text_field].apply( lambda x : len( [ x for x in x.split() if x.startswith('#') ]))
    return df


def count_usernames(df, text_field):
    '''
    count the number of users marked by a '@'
    
    inputs : 
        df : a dataframe
        text_field : the name of the text column to analyse
    
    returns :
        a copy of the dataframe df augmented by an additional column 'users_tagged'
    
    '''
    df['users_tagged'] = df[text_field].apply( lambda x : len( [ x for x in x.split() if x.startswith('@') ]))
    return df

def count_numerical_values(df, text_field):
    '''
    count the number of numerical values in a text
    
    inputs : 
        df : a dataframe
        text_field : the name of the text column to analyse
    
    returns :
        a copy of the dataframe df augmented by an additional column 'number_num_val'
    
    '''
    df['number_num_val'] = df[text_field].apply( lambda x : len( [ x for x in x.split() if x.isdigit() ]))
    return df

def count_upper(df, text_field):
    '''
    count the number of upper case words in a text
    
    inputs : 
        df : a dataframe
        text_field : the name of the text column to analyse
    
    returns :
        a copy of the dataframe df augmented by an additional column 'num_upper'
    
    '''
    df['num_upper'] = df[text_field].apply( lambda x : len( [ x for x in x.split() if x.isupper() ]))
    return df

def count_most_common_words(df, text_field, nb=10):
    '''
    count the most common words
    
    inputs : 
        df : a dataframe
        text_field : the name of the text column to analyse
    
    returns :
        a list of tuple containing the most common words and their respective number of occurences
    
    '''
    cnt = Counter()
    for text in df[text_field].values:
        for word in text.split():
            cnt[word] += 1
        
    return cnt.most_common(nb)

# Converting emojis to words
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
    return text# Converting emoticons to words    
    
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text# Example
