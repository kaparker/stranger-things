import re
import numpy as np
import io
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import PIL.ImageOps
from wordcloud import ImageColorGenerator
import string
import unicodedata
from nltk.corpus import stopwords
import nltk
from contractions import CONTRACTION_MAP
import spacy
import random

print('Loading spacy')
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
print('done')

# get data
def get_lines():
    output = []
    with open('data/season3_script.txt', 'r') as f:
        for line in f:
            output.append(line)
        f.close()
        print('Stranger things has ', len(output), 'lines')
    return output

def removetitle(text):
    return re.sub(r'.*:', '', text)

def removebrackets(text):
    return re.sub('[\(\[].*?[\)\]]', ' ', text)

def remove_accented_chars(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_special_chars(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    return re.sub(pattern, '', text)

def remove_stopwords(text):
    stopword_list = stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return ' '.join([token for token in tokens if token not in stopword_list])

def lemmatize(text):
    text = nlp(text)
    return ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    return re.sub("'", "", expanded_text)

def cleandata(df):
    # remove anything before :
    df['text_clean'] = df['text_raw'].apply(removetitle)
    # remove anything in brackets
    df['text_clean'] = df['text_clean'].apply(removebrackets)
    # remove accented chars
    df['text_clean'] = df['text_clean'].apply(remove_accented_chars)
    # remove contractions
    df['text_clean'] = df['text_clean'].apply(expand_contractions)
    # lemmatize
    print('Lemmatizing...')
    #df['text_clean'] = df['text_clean'].apply(lemmatize)
    # lowercase
    df['text_clean'] = df['text_clean'].apply(lambda x: x.lower())
    # remove special chars
    df['text_clean'] = df['text_clean'].apply(remove_special_chars)
    # remove stopwords
    df['text_clean'] = df['text_clean'].apply(remove_stopwords)
    return df

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)


def runwordcloud(image, desc):

    # based on example
    # https://github.com/keyonvafa/inaugural-wordclouds/blob/master/create_wordclouds.py
    
    # get data and clean
    print('Getting data...')
    df = pd.DataFrame(get_lines(), columns=['text_raw'])
    
    print('Cleaning data...')
    df = cleandata(df)
    print(df.head(25).to_string())

    print('Number of words', df['text_clean'].apply(lambda x: len(x.split(' '))).sum())

    # import image
    image_mask = np.array(Image.open("images/"+image+".jpeg"))
    image_colors = ImageColorGenerator(image_mask)
    # generate wordcloud
    print('Generating word cloud....')
    wc = WordCloud(background_color="black", width=400, height=400, max_words=2000, #contour_width=1, contour_color='red', 
    mask=image_mask, random_state=1).generate(' '.join(df['text_clean']))
    
    print('Making plot')
    plt.figure(figsize=(20,10))
    ypos = 650
    
    plt.style.use('dark_background')
    #plt.imshow(wc.recolor(color_func=grey_color_func))
    
    # use image colours with white background
    plt.imshow(wc.recolor(color_func=image_colors))

    plt.text(0, ypos, "Dr. K Parker - @_kaparker - https://github.com/kaparker")
    plt.axis("off")
    plt.savefig('output/wordcloud_'+image+'_'+desc+'.png', dpi=200)

runwordcloud('alexei', 's3maxwords2000')
runwordcloud('ahoy','s3maxwords2000')
runwordcloud('elemax', 's3maxwords2000')
