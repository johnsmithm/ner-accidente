# Evaluating the model
###########################
import os
import sys
import json
import spacy
import pandas as pd
sys.path.insert(0, "scripts")
from nlp.evaluate import evaluate
from nlp.evaluate import check_accuracy

real_data = os.path.join('data', 'processed', 'all_records_test_real.plk')
records_test_real = pd.read_pickle(real_data).to_records(index=False)
testing_data_real = list(records_test_real)

generated_data = os.path.join('data', 'processed', 'all_records_test_generated.plk')
records_test_generated = pd.read_pickle(generated_data).to_records(index=False)
testing_data_generated = list(records_test_generated)



nlp = spacy.load('src/models')
##########################################
results_real = evaluate(nlp, testing_data_real)
results_generated = evaluate(nlp, testing_data_generated)

data = {
    "Real": {
        "Precision": results_real['ents_p'],
        "Recall": results_real['ents_r'],
        "FScore": results_real['ents_f']
    },
    "Generated": {
        "Precision": results_generated['ents_p'],
        "Recall": results_generated['ents_r'],
        "FScore": results_generated['ents_f']
    }
}

with open("reports/metrics.json", 'w') as outfile:
    json.dump(data,outfile)

## Get accuracy by testing the whole testing_set(real + generated)
# dict_with_phrases, dict_false_positive, dict_false_negative =  check_accuracy(testing_data, model=nlp, test_set_name='All')
## Get accuracy by testing the whole real_testing_set(real only)
dict_with_phrases_real, dict_false_positive_real, dict_false_negative_real =  check_accuracy(testing_data_real, model=nlp, test_set_name='Real_data')
## Get accuracy by testing the whole generated_testing_set(generated only)
dict_with_phrases_generated, dict_false_positive_generated, dict_false_negative_generated = check_accuracy(testing_data_generated, model=nlp, test_set_name='Generated_data')

##Create dict that holds all the wrong predictions made by the model
df2 = pd.DataFrame()

df3 = pd.DataFrame()

df2['PHRASE_WRONG_real'] = pd.Series(dict_with_phrases_real)
df2['Location_false_positive_real'] = pd.Series(dict_false_positive_real)
df2['Location_false_negative_real'] = pd.Series(dict_false_negative_real)

df3['PHRASE_WRONG_generated'] = pd.Series(dict_with_phrases_generated)
df3['Location_false_positive_generated'] = pd.Series(dict_false_positive_generated)
df3['Location_false_negative_generated'] = pd.Series(dict_false_negative_generated)


# EXTRACT POSITIONS ONLY || [(112, 120, LOC_ACCIDENT)] --> [112, 120] || FROM THE LISTS
from nlp.evaluate import extract_pos
import numpy as np




df2['false_positive_poss_real'] = df2['Location_false_positive_real'].apply(extract_pos)
df2['false_negative_poss_real'] = df2['Location_false_negative_real'].apply(extract_pos)

df3['false_positive_poss_generated'] = df3['Location_false_positive_generated'].apply(extract_pos)
df3['false_negative_poss_generated'] = df3['Location_false_negative_generated'].apply(extract_pos)


from nlp.evaluate import extract_word


# axis = 0: by column = column-wise = along the rows
# axis = 1: by row = row-wise = along the columns
df2["false_positive_words_real"] = df2[["PHRASE_WRONG_real", "false_positive_poss_real"]].apply(extract_word, axis=1)
df2["false_negative_words_real"] = df2[["PHRASE_WRONG_real", "false_negative_poss_real"]].apply(extract_word, axis=1)

df3["false_positive_words_generated"] = df3[["PHRASE_WRONG_generated", "false_positive_poss_generated"]].apply(extract_word, axis=1)
df3["false_negative_words_generated"] = df3[["PHRASE_WRONG_generated", "false_negative_poss_generated"]].apply(extract_word, axis=1)


df2['false_positive_words_string_real'] = df2['false_positive_words_real'].apply(lambda x: ' '.join(x))
df2['false_negative_words_string_real'] = df2['false_negative_words_real'].apply(lambda x: ' '.join(x))

df3['false_positive_words_string_generated'] = df3['false_positive_words_generated'].apply(lambda x: ' '.join(x))
df3['false_negative_words_string_generated'] = df3['false_negative_words_generated'].apply(lambda x: ' '.join(x))


# # iterating the columns 
# for col in df2.columns:
#     print('2',col)

# for col in df3.columns: 
#     print('3',col) 

df2.drop(df2.columns.difference(['PHRASE_WRONG_real','false_positive_words_string_real','false_negative_words_string_real']), 1, inplace=True)
df3.drop(df3.columns.difference(['PHRASE_WRONG_generated','false_positive_words_string_generated','false_negative_words_string_generated']), 1, inplace=True)



# pd.set_option('display.max_rows', 100) 
print(df2)
print(df3)
df2_in_json = df2.to_json("reports/real_TYPE_I_and_II.json")
df3_in_json = df3.to_json("reports/generated_TYPE_I_and_II.json")
