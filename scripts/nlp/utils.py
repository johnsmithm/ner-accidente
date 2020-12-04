

def bars2spacy(text_bars):
    """
    input: a ||b|| c ||d|| e
    output: [(2,3,'POS'), (4,5,'POS')]
    """
    pass

def bars2y(text_bars):
    """
    input: a ||b|| c ||d|| e
    output: [0,1,0,1,0]
    """
    pass

def bars2yLetters(text_bars):
    """
    input: aa ||bb|| cb ||db|| eb
    output: ---++----++---
    """
    pass


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

