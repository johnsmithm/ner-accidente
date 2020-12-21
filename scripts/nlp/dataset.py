# all the dataset transformation to the x and y for training
import numpy as np
import pandas as pd
from random import choice
import csv, json
from itertools import zip_longest
import sys
import os
sys.path.append(".")

def preprocesare(p):
  #return p
  diacritice = [['ș','s'],['ş','s'],['î','i'],['ă','a'],['ț','t'],['â','a'],['ţ','t']]
  r = []
  for w in p.split(' '):
    if len(w)==0:
      continue
    w = w.lower()
    for f,t in diacritice:
      w=w.replace(f,t)
    r.append(w)
  return ' '.join(r)

voctext = '!~ abcdefghijklmnopqrstuvwxyz,.0123456789[]()' + ''.join([i.upper() for i in 'abcdefghijklmnopqrstuvwxyz'])
vocI = [i for i in voctext]


preprocesare('strada Decebal.Ș'), voctext


sys.path.insert(0, "../data")
print(os.listdir('.'))
#has the text from news articles about accidents + id in the ArticleLinks
with open('data/raw/generate_data/x.txt') as json_file:
    x = json.load(json_file)
a = []
hd = None
with open('data/raw/generate_data/ArticleLinks.csv') as csvfile:#has the location/street info + link to the news articles
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if hd is None:
          hd = row
          continue
        a.append({u: row[i] for i,u in enumerate(hd)})
db = []
for v,i in x:#merge text and the labels
  o = {'id':i,'text':[preprocesare(v1) for v1 in v]}
  for p in a[i].keys():#['localitate', 'location', 'street','pedestrian','decedati','sector','photo_link']:
    o[p] = preprocesare(a[i][p])
  db.append(o)

def getColumn(column, a):
  location = {}
  for i in a:
    text = preprocesare(i[column])
    if len(text)>3 and len([j for j in text if j in vocI])>len(text)/3:
      if text not in location:
        location[text]=0
      location[text] += 1
  return location

##################################
location = getColumn('location', a)
len(location), sorted([[i,j] for i,j in location.items()], key=lambda x: x[1], reverse=True)[:10]
locatitate = getColumn('localitate', a)
len(locatitate), sorted([[i,j] for i,j in locatitate.items()], key=lambda x: x[1], reverse=True)[:10]
location_trasee = getColumn('location_trasee', a)
len(location_trasee), sorted([[i,j] for i,j in location_trasee.items()], key=lambda x: x[1], reverse=True)[:20]
street = getColumn('street', a)
len(street), sorted([[i,j] for i,j in street.items()], key=lambda x: x[1], reverse=True)[:10]
obj = {#cuvinte sinonime
    'subiect1':['un taxi','un sofer','o soferita','o masina','un automobil','o motocicleta','un autobuz', 'o rutiera', 'o biclicleta','un autoturism'],
    'subiect2':['doua masini','doua automobile','doua motociclete','o masina si un autobuz','doua autoturisme'],
    'action1':['a tamponat', 'a spulberat','a accidentat', 'a lovit', 'a traumat', 'a intrat in', 'a avariat', 'a derapat','a ciocnit'],
    'action2':['a fost tamponat','a fost accidentat', 'a fost lovit', 'a fost traumat','a fost ciocnit', 'a fost spulberat'],
    'action3':['s-au tamponat', 's-au lovit', 's-au ciocnit'],
    'action4':['s-a produs','s-a intamplat', 's-a inregistrat', 'a avut loc'],
    'cum': ['mortal', 'foarte tare', 'nu grav'],
    'accident':['tragedia','accidentul','evenimentul'],
    'cand':['aseara','azi','ieri','saptamana aceasta','in aceasta dimineata','in aceasta dupa amiaza'],
    'object1':['un pieton','o fata', 'un baiat', 'o familie','o femeie','un copil','un batran','o batrana','doi pietoni','un student'],
    'object2':['o masina','un automobil','o motocicleta','un autobuz', 'o rutiera', 'o biclicleta','un gard'],
    'unde1': ['trecere de pietoni'],
    'ziua':['sambata','luni','marti','miercuri','joi','vineri','duminica'],
    'marca':['ford','mercedes','logan','audi','bmw'],
    'sofer':['sofer','soferita'],
    'autoturism':['masina','autoturism','automobil','taxi','autobuz','rutiera'],
    'duse':['au fost transportate','au fost duse'],
    'spital':['spital','urgeanta','reanimare'],
    'locuieste':['locuieste', 'este din','este locuitor'],
    'circula':['circula','mergea','se afla'],
    'murit':['pasagerul','victima','soferul','soferita'],
    'location':[i for i in location.keys()],
    'oras':[i for i in locatitate.keys()],
    'traseu':[i for i in location_trasee.keys()],
    'sector':['Buiucani','Botanica','Rascani','Telecentru','Centru'],#[i for i in location_trasee.keys()],
    'strada':[i for i in street.keys()],
    'strada1':[i for i in street.keys()],
    'ani':[str(i) for i in range(2,80)],
    'sosea':['traseu','sosea'],
    'ora':[str(i)+':'+str(j) for i in range(1,24) for j in ['00','20','30','40','50']],
    'data':[str(i)+' '+str(l) for i in range(1,31) for l in ['ianuarie','februarie','martie','aprilie','mai','decembrie']]
}
pre = ['accident grav in {oras}!']#propozitii inainte.
#propozitii dupa
more = ['potriv politiei persoana traumat a fost luat de o ambulanta si transport de urgent la','In urma impactului un taxi s-a rasturnat iar celalalt automobil',
        'Dupa produc accident sofer {marca}-ul a scos numererele']
#templates exemple accidente
samples = [
        
        'potrivit martorilor {sofer} de {ani} de ani {circula} {strada} iar {marca}-ul pe {strada1}.',
        #potriv martorilor sofer de 25 de ani circul pe strad vlaicu parcalab iar bmw-ul pe mihail kogalniceanu dup produc accident sofer bmw-ul a scos numer
        '{object1} {duse} la {spital} in urma unu accident {location}.',
        #patru persoan au fost transport la spital de urgenta in urma unu accident pe soseau balcani intersect cu strad deleanu
        '{ziua} {data} in jur ore {ora} {location} {sofer} unui {autoturism} de model {marca} {action1} {object1}.',
        #sambata 28 octombrie in jur ore 19:00 pe strad vas alecsandr din or cahul sofer unu autoturist de model „ford” a tampon un barbat
        '{subiect2} {action3} {cand} {location}.',
        '{subiect2} {action3} in jurul orei {ora} {location}.',
        '{subiect2} {action3} in jurul orei {ora} pe sosea {traseu}.',
        '{ziua} {data} {subiect2} {action3} {cand} {location}.',
        #dou masin s-au ciocnit asear la intersect bulevard daci si traian din capitala in urma impactului un tax s-a rasturnat iar celalalt automobil
        '{accident} {action4} {cand} {location}.',
        '{accident} {action4} {cand} la intersectia dintre {strada} si {strada1}.',
        '{accident} {action4} la ora {ora} la intersectia dintre {strada} si {strada1}.',
        #accident s-a intampl in aceast dimineata pe soseau munc din capitala
        "{subiect1} {action1} {object1} {unde1} {location}.",
        #un sofer a tamponat o femeie p trec de pietoni la intersect bd decebal cu strad trandafirilor potriv pol
        "{object1} {action2} {cum} {cand} {subiect1} {accident} {action3} {location} in jur orei {ora}.",
        #un pieton a fost spulber mortal asear de un taxi accident s-a produs pe soseau munc din capitala in jur ore 21:00
]
#exemple negative, nu despre accidente
false = [
        'in zona s-au creat ambuteiaj kilometrice, coloana de masini blocheaza trafic de la {strada}',
        #in zon s-au creat ambuteiaj kilometrice coloan de masin bloc trafic de la cal basarab pan in regiun circului
        '{object1} {locuieste} {oras}',
        '{object1} care a fost accidentat {locuieste} {oras}',
        '{sofer} care a creat accidentul {locuieste} {oras}',
        '{strada} chisinau md-{ani}8 republica moldova tel: (022) {ani}-{ani}-{ani} mob: {ani} {ani} {ani}',
        '{sofer} {locuieste} {strada} {oras}',
        '{murit} de {ani} de ani a fost internata in sectia reanimare din spital raional {oras} cu diferite traume',
        # decedat iar pasagerul de 28 de ani a fost intern in sect reanim din spital raional telenesti cu difer traume
        'in zona se circula cu dificulte pe {strada} direct spre {strada1} astfel serviciul ”infotrafic” recomanda conduc auto:',
        #in zona se circul cu dificult pe strismail direct spre strcal basarabiei astfel servic ”infotrafic” recomand conduc auto:
]
# return samples, false
########################


def rd():
  return np.random.rand()
def getTemplate(template):
  keys =  [i.split('}')[0] for i in template.split('{')[1:]]
  y = []
  for key in keys:
    if key in obj:
      #print(obj[key])
      w = choice(obj[key])
      if key in ['location','strada','strada1','oras','traseu']:
        y.append([template.find('{'+key+'}'), len(w)])
      template=template.replace('{'+key+'}',w)
    else:
      print(1111,key)
  return template, y

def createSentence(samples):
  text = ''
  if rd()>0.8:
   text+= getTemplate(choice(pre))[0]
  
  template = choice(samples)
  #print(template)
  template, y = getTemplate(template)
  text=''+template

  if rd()>0.8:
    template = choice(more)
    text+= ' '+getTemplate(template)[0]
  Y = [];
  l = 0
  for i,j in y:
    Y.extend('-'*(i-l))
    Y.extend('+'*(j))
    l += j+i
  Y.extend('-'*(len(text)-len(Y)))
  return text, ''.join(Y)


def create_many_Sentences(number_of_sentences = 1, last_index=0, create_from_what = samples):
    texts = []
    marks = []
    indexes = []
    for index, i in enumerate(range(0,number_of_sentences)):
        index = index + last_index
        x,y = createSentence(samples)
        # print(x)
        # print(y)
        indexes.append(index)
        texts.append(x)
        marks.append(y)

    data = {'id': indexes,
           'y': 1,
           'text_unprocessed': texts,
           'marks': marks,
           'com': None}
    return data

def get_id_of_last_elem_in_dataset(path):
  with open(path, 'r') as real_data:
      data = real_data.readlines()
      last_row = data[-1].split('\t') # Update for tab delimit.
      last_row = ''.join(last_row).split(',')
      # print(last_row)
      cell_with_last_id = last_row[0]
      cell_with_last_id = int(cell_with_last_id)
      # print(cell_with_last_id)
      return cell_with_last_id

def turn_data_to_df(data_to_create_from):
    df = pd.DataFrame(data_to_create_from)
    df = df.astype({"y": int})
    return df

def save_df_to_csv(df,path_where_to_save):
    df.to_csv(index=False)

def join_2_df_into_1_drop_uneeded(df1,df2):
    #df1 is assumed to be the column that holds the real values
    #df2 is the one that holds the generated data
    if 'text_unprocessed' in df2.columns:
        df2 = df2.drop(columns=['text_unprocessed','marks'])
        frames = [df1, df2]
        df3 = pd.concat(frames, ignore_index=True, sort=True)
        df3.reset_index(drop=True)
    return df3


### START PROCESSING OF THE GENERATED PHRASES
def get_index_start_finish(x,y):
    # print(x)
    # print(y)
    try:
        start_keyword_index = []
        end_keyword_index = []
        for index,(f, b) in enumerate(zip(x,y)):
            index = index + 1
            # print(index,(f, b))
            if y[index] == '-' and y[index+1] == '+':
                # print('START_INDEX',index+1)
                start_keyword_index.append(index+1)
            elif y[index] == '+' and y[index+1] == '-':
                # print('END_INDEX',index+1)
                end_keyword_index.append(index+1)
    except:
        pass
    return start_keyword_index, end_keyword_index


def transform_keywords(start,end):
    enteties = []
    for f, b in zip(start,end):
        ent =[f,b]
        # print(ent)
        enteties.append(ent)
    # print('enteties',enteties)
    return enteties


def check_if_exists_in_list_of_lists(elem_to_check, list_of_lists):
    check_status = bool
    if any(elem_to_check in sublist for sublist in list_of_lists):
        check_status = True
    else:
        check_status = False
    return check_status

def get_space_in_enteties(ent_ranges, phrase):
    spaces_in_ent = []
    for ent in ent_ranges:
        # print(phrase[ent[0]:ent[1]])
        for i in range(ent[0],ent[1]):
            if phrase[i] == ' ':
                spaces_in_ent.append(i)
        for i in spaces_in_ent:
            check = check_if_exists_in_list_of_lists(i,ent_ranges)
            if check:
                spaces_in_ent.remove(i)
                # print('SPACE REPEATING IN START',i)
    # print('SPACES',spaces_in_ent)
    return spaces_in_ent


def repalce_in_phrase(index_start_finish,index_spaces, phrase):
#     print(index_start_finish)
#     print(phrase)
    
    phrase_in_list = list(phrase)
    for point in index_spaces:
#         print('index_spaces', point)
        phrase_in_list[point] = '|| ||'
        
    for index,point in enumerate(index_start_finish):
        index = index+1
#         print('index_start_finish',index,point)
#         print(phrase[point])
        if index%2 == 0:
            phrase_in_list[point] = '|| '
        if index%2 == 1:
            phrase_in_list[point-1] = ' ||'

    final_phrase = ''.join(phrase_in_list)
#     print(final_phrase)
    return final_phrase


def yLetters2bars(text, yLetters):
    """
    input: 
    - aa bb cb db eb
    - ---++----++---
    output: aa ||bb|| cb ||db|| eb
    """
    # print('TEXT_LEN:',len(text),'MARKS_LEN:',len(yLetters))
    
    start_index,end_index = get_index_start_finish(text,yLetters)
    # print(start_index,end_index)
    index_to_replace_with_bars = transform_keywords(start_index,end_index)
    # print(index_to_replace_with_bars)
    spaces_in_ent = get_space_in_enteties(index_to_replace_with_bars, text)
    # print(spaces_in_ent)
    joined_index_to_replace_with_bars =  [y for x in index_to_replace_with_bars for y in x]
    final_phrase = repalce_in_phrase(joined_index_to_replace_with_bars,spaces_in_ent, text)
    # print(final_phrase)
    return final_phrase

### END PROCESSING OF THE GENERATED PHRASES

def devide_enteties(ent_index,spaces_in_ent):
    # print('ent_index',ent_index)
    ent_possitions = []
    all_ent_pos = []
    end_of_ent = []
    for ent_pos in ent_index:
#         print('ent',ent_pos)
        all_ent_pos.extend([ent_pos[0],ent_pos[1]])
        spaces_per_ent = []
        for space in spaces_in_ent:
            if space in range(ent_pos[0],ent_pos[1]):
                spaces_per_ent.append(space)
#         print('spaces_per_ent',spaces_per_ent)
        for space in spaces_per_ent:
#             end_of_ent.append(space-1)
            end_of_ent.append(space+1)
            
#     print('end_of_ent',end_of_ent)
    ent_possitions = end_of_ent+all_ent_pos+spaces_in_ent
#     print('ent_possitions',ent_possitions)
    ent_possitions.sort()
#     print('ent_possitions',ent_possitions)
    return ent_possitions


def pair_list(list_):
    return[list_[i:i+2] for i in range(0, len(list_), 2)]

def spacy_transform(all_poss,text):
    # print(text[all_poss[0]:all_poss[-1]])
    spacy_format_ents = []
    pairs = pair_list(all_poss)
    for one_pair in pairs:
        # print(text[one_pair[0]:one_pair[1]])
        formated_ent = (one_pair[0],one_pair[1],'POS')
        # print(formated_ent)
        spacy_format_ents.append(formated_ent)
    # print(spacy_format_ents)
    return spacy_format_ents

def yLetters2spacy(text, yLetters):
    """
    input: 
    - aa bb cb db eb
    - ---++----++---
    output: [(3,5,'POS'), (8,9,'POS')]
    """
#     print(text)
#     print(yLetters)
    start_index,end_index = get_index_start_finish(text,yLetters)
#     print('start_index,end_index',start_index,end_index)
    index_to_replace_with_bars = transform_keywords(start_index,end_index)
#     print('index_to_replace_with_bars',index_to_replace_with_bars)
    spaces_in_ent = get_space_in_enteties(index_to_replace_with_bars, text)
#     print('spaces_in_ent',spaces_in_ent)
    all_ent_poss = devide_enteties(index_to_replace_with_bars,spaces_in_ent)
#     print('all_ent_poss',all_ent_poss)
    all_formater_ents = spacy_transform(all_ent_poss,text)
    return all_formater_ents


def read_csv_to_df(path_to_csv):    
    #use read_csv
    test = pd.read_csv(path_to_csv)
    df = test
    return df


def generate_data(templates):
    pass

def join_generated_and_real(df_generated, df_real):
    #use concat
    return joined_data

def df_to_scipy_db(df):
    #use apply
    pass