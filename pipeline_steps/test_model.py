# Evaluating the model
###########################
import os
import sys
import json
import spacy
import pandas as pd
sys.path.insert(0, "scripts")
from nlp.evaluate import evaluate


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
    "REAL": {
        "Precision": results_real['ents_p'],
        "Recall": results_real['ents_r'],
        "FScore": results_real['ents_f']
    },
    "GENERATED": {
        "Precision": results_generated['ents_p'],
        "Recall": results_generated['ents_r'],
        "FScore": results_generated['ents_f']
    }
}

with open("reports/metrics.json", 'w') as outfile:
    json.dump(data,outfile)