# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.2.4
"""

import random
import warnings
from pathlib import Path
import spacy
import pandas as pd
import numpy as np
from spacy.util import minibatch, compounding

# Use built-in "ner" pipeline components
nlp = spacy.load("ro_core_news_lg")
ner = nlp.get_pipe("ner")

def main(model=None,TRAIN_DATA=None , output_dir=None, n_iter=100):

    # add labelsmodel
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    print("Started Training")
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

if __name__ == "__main__":



    import os
    import sys
    sys.path.insert(0, "src")
    sys.path.insert(0, "scripts")

    path = os.path.join('data', 'processed', 'data_processed.plk')
    df = pd.read_pickle(path)

    df['text_no_sw_no_bars'] = df['text_no_sw'].str.replace('|','')
    
    from nlp.pptext import split_and_save_to_new_column
    holding_dict = split_and_save_to_new_column(df['text_no_sw'])


    ## Insert our data in spacy training format into the df column 'Training_format'
    if 'Entities_position' in df.columns:
        df = df.drop(columns='Entities_position')

    # Assign to the previously created column the dictionaries with the values
    df['Entities_position'] = pd.Series(holding_dict)
    
    print(df)
    ## Deviding the Training-Set from the testing set
    # join half of the real phrases with half of the generated ones for both training and testing
    records_train = df[['text_no_sw_no_bars','Entities_position']].iloc[np.r_[0:80, 155:503]].to_records(index=False)
    train_data = list(records_train)
    
    records_test_real = df[['text_no_sw_no_bars','Entities_position']][80:155].to_records(index=False)
    testing_data_real = list(records_test_real)

    records_test_generated = df[['text_no_sw_no_bars','Entities_position']][155:503].to_records(index=False)
    testing_data_generated = list(records_test_generated)
    
    # MAIN TRAINING FUNCTION
    main(model='ro_core_news_lg', TRAIN_DATA=train_data, output_dir='src/models/', n_iter=1)

############

    # Evaluating the model
    ###########################
    from nlp.evaluate import evaluate
    nlp = spacy.load('src/models')
    ##########################################
    results = evaluate(nlp, testing_data_real)
    results_real = (f"-------------------------\nFor real_examples\nprecision is: {results['ents_p']}\nrecall is: {results['ents_r']}\nfscore is: {results['ents_f']}")
    results = evaluate(nlp, testing_data_generated)
    results_generated = (f"-------------------------\nFor generated_examples\nprecision is: {results['ents_p']}\nrecall is: {results['ents_r']}\nfscore is: {results['ents_f']}")

    with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(results_real) + "\n" + "Accuracy: " + str(results_generated) + "\n")