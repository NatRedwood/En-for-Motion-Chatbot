import os
import pandas as pd
import numpy as np

def preprocess(file):
    path = file
    data = pd.read_csv(path, sep = ';')
    hos = []
    for i in range(len(data.emotion)):
        if data['emotion'][i] in ['joy', 'love']:
            hos.append(1) # happy is 1
        else:
            hos.append(0) # sadness, anger, fear, surprise is 0
    data['hos'] = hos
    return data


def postprocessor(preds, predstr):
  range = predstr.max()-predstr.min()
  print(range)
  probab = []
  for i in preds:
    probab.append((i - predstr.min()) * 100 / range)
    #print(norm_preds)
  print(probab)
  return np.mean(probab)

def hms(seconds):
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 3600 % 60)
    return f'{h}:{m}:{s}'