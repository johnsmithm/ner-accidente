import spacy
from spacy import displacy

# preprocessing function

def clean_data(dataframe_column):
    print(f'Started cleaning: the dataframe column from all the special characters\n')
    dataframe_column = dataframe_column.str.replace('\xa0',' ')
    dataframe_column = dataframe_column.str.replace('\|\|-\|\|','')

    spec_chars = ["!",'"',"#","%","&","'","(",")",
                  "*","+",",",".","/",";","<",
                  "=",">","?","@","[","\\","]","^","_",
                  "`","{","}","~","–"]
    for char in spec_chars:
        dataframe_column = dataframe_column.str.replace(char, ' ')

    dataframe_column = dataframe_column.str.replace(r'^\s+|\s+$', '')
    dataframe_column = dataframe_column.str.replace('    ',' ')
    dataframe_column = dataframe_column.str.replace('  ',' ')
    dataframe_column = dataframe_column.str.replace('  ',' ')
    print(dataframe_column[13])

    dataframe_column_clean = dataframe_column

    all_corpus_text = []
    for phrases in dataframe_column:
        phrases = phrases.split(" ")
        for words in phrases:
            all_corpus_text.append(words)
    return all_corpus_text , dataframe_column_clean


# pass it a dataframe column and it will go over them to extract all the entities that it has and displays them
def detect_entities_in_dataframe(dataframe_column, model_path):
    for sentence in dataframe_column:
        ## print named entities in phrases
        nlp = spacy.load(model_path)
        sentence_nlp = nlp(sentence)
        print([(word, word.ent_type_) for word in sentence_nlp if word.ent_type_])
        # # visualize named entities
        displacy.render(sentence_nlp, style='ent', jupyter=True)

# pass it 2 lists of words and it will create a new list with all the words that appear in both lists and how many times
def get_words_appearing_in_both(marked_words, unmarked_words):
    words_that_appear_in_both = []
    for word in unmarked_words:
        if word in marked_words:
            words_that_appear_in_both.append(word)
    return words_that_appear_in_both

def get_how_many_entities_and_non_entities_per_phrase(dataframe):
    all_entities = {}
    all_non_entities = {}
    # the start and end positions of the entities in pohrases 
    all_entities_positions = {}
    # Dic that will hold the number of words that are in "all_entities" and "all_non_entities"
    num_all_entities = {}
    num_all_non_entities = {}
    sentence_index = -1
    for sentence in dataframe:
        entities = []
        non_entities = []
        sentence_index += 1
        for i in sentence.split():
            i_processed = i.replace("|","")
            if "|" in i:
                entities.append(i_processed)
            else: 
                non_entities.append(i_processed)
#         print(f'IN ROW:{sentence_index} there are {len(entities)}={entities} entities in this phrase and {len(non_entities)}={non_entities} non_entities')
        
        all_entities[sentence_index] = entities
        all_non_entities[sentence_index] = non_entities
        
        num_all_entities[sentence_index] = int(len(entities))
        num_all_non_entities[sentence_index] = int(len(non_entities))
        
    return all_entities, all_non_entities, num_all_entities, num_all_non_entities



def change_to_training_format(passed_data):
    last_len = 0
    position = []
    entities = []

    for i in passed_data:
        i_processed = i.replace("||","")
        if "|" in i:
            start = last_len
            finish = last_len + len(i_processed)
            last_len += len(i_processed) + 1

            entities.append((start, finish, 'LOC_ACCIDENT'))
            position.append(start)
            position.append(finish)
        else:
            last_len += len(i_processed) + 1
    return entities

def split_and_save_to_new_column(df_column):
# this dict will be populated with "entities_positions_dict" for each row
    holding_dict = {}
    row_index = 0
    for row in df_column:
        # all the enteties per row will be held in this dic 
        entities_positions_dict = {}

        split = row.split(" ")

        #Data cleaning some symbols
        cleaned_data_no_bars = row
        cleaned_data_no_bars = cleaned_data_no_bars.replace('||', " ")
        cleaned_data_no_bars = cleaned_data_no_bars.replace('   ', " ")
        cleaned_data_no_bars = cleaned_data_no_bars.replace('  ', " ")

        entities_position = change_to_training_format(split)
        # Append
        entities_positions_dict['entities'] = entities_position
        holding_dict[row_index] = entities_positions_dict
        row_index +=1
    return holding_dict

def split_in_ent_and_non_ent(df_column):
    all_marked_text = []
    all_unmarked_text = []
    for phrases in df_column:
    #     print('1' + phrases)
        phrases = phrases.split(" ")
        for words in phrases:
    #         print(phrases)
            i_processed = words.replace("|","")
            if "|" in words:
                all_marked_text.append(i_processed)
            else:
                all_unmarked_text.append(words)
    return all_marked_text, all_unmarked_text


