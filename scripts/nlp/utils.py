import sys
sys.path.append(".")

def bars2spacy(text_bars):
    """
    input: a ||b|| c ||d|| e
    output: [(2,3,'LOC_ACCIDENT'), (4,5,'LOC_ACCIDENT')]
    """
    split_phrase = text_bars.split()
    start_len = 0
    enteties_per_phrase = []
    for word in split_phrase:
        x = word.startswith("||")
        if x: # IF X IS TRUE
            START_ENT_POS = start_len
            FINISH_ENT_POS = start_len + (len(word)-4) # -4 so that we take into account the bars
            enteties_per_phrase.append((START_ENT_POS,FINISH_ENT_POS,'LOC_ACCIDENT'))
            start_len += (len(word)-3) # -3 because of the space before the word
        else:
            start_len += (len(word)+1)

    return enteties_per_phrase

def bars2y(text_bars):
    """
    input: a ||b|| c ||d|| e
    output: [0,1,0,1,0]
    """
    split_phrase = text_bars.split()
    bars2y = []
    for word in split_phrase:
        x = word.startswith("||")
        if x: # IF X IS TRUE
            bars2y.append(1)
        else:
            bars2y.append(0)

    return bars2y

def bars2yLetters(text_bars):
    """
    input: aa ||bb|| cb ||db|| eb
    output: ---++----++---
    """
    output_str = []
    text_bars_clean = text_bars.replace('||', '')
    text_bars_clean_list = list(text_bars_clean)
    position = bars2spacy(text_bars)
    indexes = []
    for index, pos in enumerate(position):
        indexes_pf = []
        indexes_pf.extend((position[index][0],position[index][1]))
        indexes.append(indexes_pf)
    for place in indexes:
        for i in range((place[0]),(place[1])):
            text_bars_clean_list[i] = '+'
    for index,letter in enumerate(text_bars_clean_list):
        if text_bars_clean_list[index] == '+':
            pass
        else:
            text_bars_clean_list[index] = '-'    
    text_bars_clean = ''.join(text_bars_clean_list)
    return text_bars_clean


def yLetters2bars(text, yLetters):
    """
    input: 
    - aa bb cb db eb
    - ---++----++---
    output: aa ||bb|| cb ||db|| eb
    """
    pass

def yLetters2spacy(text, yLetters):
    """
    input: 
    - aa bb cb db eb
    - ---++----++---
    output: [(3,5,'POS'), (8,9,'POS')]
    """
    pass

def y2bars(y, text):
    """
    input: 
    - [0,1,0,1,0] -> BARS2Y
    - a b c d e -> text
    output: a ||b|| c ||d|| e
    """
    text = text.split()
    for index,word in enumerate(zip(y,text)):
        if word[0] == 1:
            text[index] = '||'+ text[index] + '||'
    text = ' '.join(text)
    return text

def spacy2bars(entities, text):
    """
    input: 
    - [(2,3,'POS'), (4,5,'POS')]
    - a b c d e
    
    output: aa ||bb|| cc ||dd|| ee
    """
    all_ent = []
    for ent in entities:
        ent_pos = []
        ent_pos.extend([ent[0],ent[1]])
        all_ent.append(ent_pos)
    for index,ent in enumerate(all_ent):
        start_word = all_ent[index][0]
        end_word = all_ent[index][1]
        # Add additional numbers to the index so that
        add_to_index = index*4
        # it takes into account the added || to phrase
        old = text[start_word+add_to_index:end_word+add_to_index]
        new = '||' + text[start_word+add_to_index:end_word+add_to_index] + '||'
        text = text.replace(old, new)
        
    return text

def spacy2y(entities, text):
    """
    input: 
    - [(3,5,'POS'), (9,11,'POS')]
    - aa bb cc dd ee
    output: [0,1,0,1,0]
    """
#     print(entities, text)
    marked_text = spacy2bars(entities, text)
#     print(marked_text)
    ent_or_nonent = bars2y(marked_text)
#     print(ent_or_nonent)
    return ent_or_nonent

from nlp.pptext import change_to_training_format 

def y2spacy(y, text):
    """
    input: 
    - [0,1,0,1,0]
    - aa bb cc dd ee
    output: [(3,5,'POS'), (9,11,'POS')]
    """
    marked_text = y2bars(y, text)
    # print(marked_text)
#     split = marked_text.split(" ")
    spacy_format_ent = change_to_training_format(marked_text)
    # print(text[result[0][0]:result[0][1]])
    print(spacy_format_ent)
    return spacy_format_ent

def calculate_recall(y_pred, y_true):
    """
    input: 
    - [0,1,0,1,0]
    - [0,1,0,1,0]
    output: 1
    
    
    input: 
    - [0,1,0,1,0]
    - [0,1,0,0,0]
    output: 1
    """
    pass

def calculate_acc(y_pred, y_true):
    """
    input: 
    - [0,1,0,1,0]
    - [0,1,0,1,0]
    output: 1
    
    
    input: 
    - [0,1,0,1,0]
    - [0,1,0,0,0]
    output: 1
    """
    pass

def calculate_acc_ignore0(y_pred, y_true):
    """calculate the accuracy just for the non-zero tokens
    input: 
    - [0,1,0,1,0]
    - [0,1,0,1,0]
    output: 1
    
    
    input: 
    - [0,1,0,1,0]
    - [0,1,0,0,0]
    output: 1
    """
    pass

def calculate_confusion_matrix(y_pred, y_true):
    """
    input: 
    - [0,1,0,1,0]
    - [0,1,0,1,0]
    output: 1
    
    
    input: 
    - [0,1,0,1,0]
    - [0,1,0,0,0]
    output: 1
    """
    pass

def calculate_precision(y_pred, y_true):
    """
    input: 
    - [0,1,0,1,0]
    - [0,1,0,1,0]
    output: 1
    
    
    input: 
    - [0,1,0,1,0]
    - [0,1,0,0,0]
    output: 1
    """
    pass

