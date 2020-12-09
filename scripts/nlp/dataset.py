# all the dataset transformation to the x and y for training
import numpy as np
from random import choice
import csv, json

voctext = '!~ abcdefghijklmnopqrstuvwxyz,.0123456789[]()' + ''.join([i.upper() for i in 'abcdefghijklmnopqrstuvwxyz'])
vocI = [i for i in voctext]
voc = {i:j for j,i in enumerate(voctext) }
def preprocesare(p):
  #return p
  diacritice = [['ș','s'],['ş','s'],['î','i'],['ă','a'],['ț','t'],['â','a'],['ţ','t']]
  r = []
  for w in p.split(' '):
    if len(w)==0:
      continue
    w = w.lower()
    for f,t in diacritice:
      # f,t = e
      # f = e[0]
      # t = e[1]
      w=w.replace(f,t)
    # for f,t in diacritice:
    #   w=w.replace(f.upper(),t.upper())
    r.append(w)
  return ' '.join(r)

preprocesare('strada Decebal.Ș'), voctext


with open('../data/raw/generate_data/x.txt') as json_file:#has the text from news articles about accidents + id in the ArticleLinks
    x = json.load(json_file)
a = []
hd = None
with open('../data/raw/generate_data/ArticleLinks.csv') as csvfile:#has the location/street info + link to the news articles
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






def readDF(path):    
    #use read_csv
    df = 'readfile'
    return df


def generate_data(templates):
    pass

def join_generated_and_real(df_generated, df_real):
    #use concat
    return joined_data

def df_to_scipy_db(df):
    #use apply
    pass