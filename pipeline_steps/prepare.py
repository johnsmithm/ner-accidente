# Standard python packages
import os
import sys
import string
# # for natural language processing: named entity recognition
# import spacy
# from spacy.displacy.render import EntityRenderer
# from IPython.core.display import display, HTML

# Other package imports
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.pardir, os.pardir))
data_folder = os.path.join(project_root, 'data')
data_folder_raw = os.path.join(data_folder, 'raw')
src_folder = os.path.join(project_root, 'src')

# Explicitly set path so don't need to run setup.py - if we have multiple copies of 
# the code we would otherwise need to setup a seperate environment for each to
# ensure the code pointers are correct.
sys.path.insert(0, "../src")
sys.path.insert(0, "../scripts")


# Phrases from which new ones will be generated
from nlp.dataset import samples
# Function that creates training_data accepts samples from which
from nlp.dataset import create_many_Sentences
from nlp.dataset import turn_data_to_df
from nlp.dataset import get_id_of_last_elem_in_dataset
from nlp.dataset import yLetters2bars
from nlp.dataset import join_2_df_into_1_drop_uneeded

path = os.path.join('../data', 'raw', 'raw_data.csv')

df = pd.read_csv(path)
# Gets the last ID from the real_examples and returns it
last_id = get_id_of_last_elem_in_dataset(path)
# Created a set numer of generated examples and returns the data
data = create_many_Sentences(number_of_sentences=200, last_index=last_id, create_from_what=samples)
# Converts the returned data dict into a df
df2 = turn_data_to_df(data)

# if 'com' in df.columns:
#     df = df.drop(columns='com')
# df = df[df.isna().any(axis=1)]
# print(df)

'''
Use np.vectorize(NAME_OF_FUNCTION)(df2['Col_name1'], df2['Col_name2'])
This allows up to pass arguments to a function that accepts multple arguments
'''
if 'text' in df2.columns:
    df = df2.drop(columns='marked_text')
df2['text'] = np.vectorize(yLetters2bars)(df2['text_unprocessed'], df2['marks'])


# if 'com' in df.columns:
#     df2 = df2.drop(columns='com')
# df2 = df2[df.isna().any(axis=1)]
# print(df2)


# Join our Real Examples to our generated examples
df3 = join_2_df_into_1_drop_uneeded(df,df2)
df = df3

# TIME TO CLEAN THE TEXT
import spacy
from spacy import displacy
# install romanian models
# !python3 -m spacy download ro_core_news_lg


nlp = spacy.load("ro_core_news_lg")
all_stopwords = nlp.Defaults.stop_words
# lower the text so that it all is in lowercase
df['text'] = df['text'].str.lower()
# # remove all stop words that appear in the romanian text corpus
df['text_no_sw'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (all_stopwords)]))



# Clean the dataframe column with the name text and visualize what are the main words that the model
# should take as the main features
from nlp.pptext import clean_data

all_corpus_text, dataframe_column_clean = clean_data(df['text'])

# Save the current DF into a .pkl(pickle) data format so that each cell in the DF will maintain its data-type
# to later use for analysis
df.to_pickle('../data/processed/data_processed.plk', protocol=3)

