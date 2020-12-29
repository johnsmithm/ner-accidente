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