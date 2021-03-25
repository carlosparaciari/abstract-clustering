import re

# clean the text and use lemmatizer
def clean_text_lemmatize(item,lemmatizer,stopwords):
    
    # remove latex equations
    item = re.sub('\$+.*?\$+','',item)
    
    # tokenize and remove punctuation
    item = re.findall('[a-zA-Z0-9]+',item)
    
    # lowecase everything
    item = [word.lower() for word in item]
    
    # remove english stopwords
    item = [word for word in item if word not in stopwords]
    
    # lemmatize the words
    item = [lemmatizer.lemmatize(word) for word in item]
    
    return item