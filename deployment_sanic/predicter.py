import spacy
from pre_processing import preprocesare
# from import dataset.preprocesare
nlp = spacy.load('../src/models/')


#Spacy stuff
def predict_ent(text_to_extract_from):
    text_to_extract_from_processed = preprocesare(text_to_extract_from)

    doc = nlp(text_to_extract_from_processed)

    ets_list = []
    for ent in doc.ents:
        # print(ent.text, ent.start_char, ent.end_char, ent.label_)
        tuple_ent =(ent.text, ent.label_) 
        ets_list.append(tuple_ent)
    return ets_list

# predict_ent("Totul s-a întâmplat în data de 27 octombrie pe o stradă din municipiul hincesti")