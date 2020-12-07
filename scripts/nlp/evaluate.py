# all the evaluation related function like: accuracy calc, false positive etc
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer

def evaluate(nlp, examples, ent='LOC_ACCIDENT'):
    scorer = Scorer()
    for input_, annot in examples:
        text_entities = []
        for entity in annot.get('entities'):
            if ent in entity:
                text_entities.append(entity)
        doc_gold_text = nlp.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=text_entities)
        pred_value = nlp(input_)
        scorer.score(pred_value, gold)
    return scorer.scores


def calculate_false_negatives(y_true, y_pred):
    pass

def check_accuracy(test_list, model, test_set_name='PLACEHOLDER'):
    correct = 0
    wrong = 0

    dict_phrase_issue = {}
    dict_issue_location = {} 
    dict_issue_not_found = {}
    for i in range(len(test_list)):
        ent_model_results = []
        phrase_text = model(test_list[i][0])
        entities_localtion = test_list[i][1]['entities']
        for ent in phrase_text.ents:
            tuple_ent =(ent.start_char, ent.end_char, ent.label_)
            ent_model_results.append(tuple_ent)
        if entities_localtion == ent_model_results:

            correct += 1
        else:
            # False positive
            wrong_guess = list(set(ent_model_results).difference(entities_localtion))
            # False negative
            didnt_find = list(set(entities_localtion).difference(ent_model_results))
            dict_phrase_issue[wrong] = test_list[i][0]
            dict_issue_location[wrong] = wrong_guess
            dict_issue_not_found[wrong] = didnt_find

            wrong += 1
    print(f'{correct} "<= correct | {test_set_name} | wrong =>", {wrong}')
    print(f"Guessed: {correct} from {len(test_list)} which is {(correct/len(test_list))*100}%")
    print(50*'-')
    return dict_phrase_issue, dict_issue_location, dict_issue_not_found

""" Extract the possition 2,3 and 4,5 from the  [(2,3,'POS'), (4,5,'POS')]"""
def extract_pos(column):
    single_positions = []
    if len(column) == 1:
        column = [column[0][0],column[0][1]]
        single_positions.append(column)
        column = single_positions
    elif len(column) > 1:
        multiple_positions = []
        for pos in range(len(column)):
            poss = [column[pos][0],column[pos][1]]
            multiple_positions.append(poss)
        column = multiple_positions
    return column

""" Extract the words based on their position within the string """
def extract_word(position):    
    whole_phrase = position[0]
    word_position = position[1]
    extracted_word_list = []
    if len(word_position) == 1:
        extracted_word = whole_phrase[word_position[0][0]:word_position[0][1]]
        extracted_word_list.append(extracted_word)
        position = extracted_word_list
    elif len(word_position) > 1:
        multiple_words = []
        for lenght in range(len(word_position)):
            extracted_word = whole_phrase[word_position[lenght][0]:word_position[lenght][1]]
            multiple_words.append(extracted_word)
        position = multiple_words
    else:
        position = []
    return position