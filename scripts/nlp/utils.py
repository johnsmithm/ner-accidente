

def bars2spacy(text_bars):
    """
    input: a ||b|| c ||d|| e
    output: [(2,3,'LOC_ACCIDENT'), (4,5,'LOC_ACCIDENT')]
    """
    split_phrase = text_bars.split()
    START_LEN = 0
    ENTITIES_PER_PHRASE = []
    for word in split_phrase:
        x = word.startswith("||")
        if x: # IF X IS TRUE
            START_ENT_POS = START_LEN
            FINISH_ENT_POS = START_LEN + (len(word)-4) # -4 so that we take into account the bars
            ENTITIES_PER_PHRASE.append((START_ENT_POS,FINISH_ENT_POS,'LOC_ACCIDENT'))
            START_LEN += (len(word)-3) # -3 because of the space before the word
        else:
            START_LEN += (len(word)+1)

    return ENTITIES_PER_PHRASE

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
    - [0,1,0,1,0]
    - a b c d e
    output: a ||b|| c ||d|| e
    """
    pass

def spacy2bars(entities, text):
    """
    input: 
    - [(2,3,'POS'), (4,5,'POS')]
    - a b c d e
    output: a ||b|| c ||d|| e
    """
    pass

def spacy2y(entities, text):
    """
    input: 
    - [(2,3,'POS'), (4,5,'POS')]
    - a b c d e
    output: [0,1,0,1,0]
    """
    pass

def y2spacy(text, y):
    """
    input: 
    - [0,1,0,1,0]
    - a b c d e
    output: [(2,3,'POS'), (4,5,'POS')]
    """
    pass

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

