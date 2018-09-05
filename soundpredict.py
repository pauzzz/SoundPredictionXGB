import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import os
from librosa.display import waveplot
import librosa
import glob
%pylab inline

data, sampling_rate=librosa.load('H:/train/Train/2022.wav')

plt.figure(figsize=(12,4))
waveplot(data, sr=sampling_rate)

train=pd.read_csv('H:/train/train.csv')
test=pd.read_csv('H:/test/test.csv')
train.Class.value_counts()
train.head()


#LOAD AUDIO FILES

def parserTrain(row):
  #function to load audio files and extract important features
  file_name=os.path.join(os.path.abspath('H:/train/Train'),str(row.ID)+'.wav')
  
  #handle exception to check if there's a file which isn't corrupted
  try:
    #kaiser_fast is a technique for fast extraction
    X, sample_rate=librosa.load(file_name, res_type='kaiser_fast')
    #extract mfcc feature data
    mfccs= np.mean(librosa.feature.mfcc(y=X, sr=sampling_rate, n_mfcc=40).T, axis=0)
  except Exception as e:
    print('Error encountered while parsing file:' +file_name)
    return None, None
  feature=mfccs
  label=row.Class
  return [feature, label]

def parserTest(row):
  #function to load audio files and extract important features
  file_name=os.path.join(os.path.abspath('H:/test/Test'),str(row.ID)+'.wav')
  
  #handle exception to check if there's a file which isn't corrupted
  try:
    #kaiser_fast is a technique for fast extraction
    X, sample_rate=librosa.load(file_name, res_type='kaiser_fast')
    #extract mfcc feature data
    mfccs= np.mean(librosa.feature.mfcc(y=X, sr=sampling_rate, n_mfcc=40).T, axis=0)
  except Exception as e:
    print('Error encountered while parsing file:' +file_name)
    return None, None
  feature=mfccs
  label=None
  return [feature, label]

#parse Train data
temp=train.apply(parserTrain,axis=1)


tempo=pd.DataFrame(temp)
tempo['a.x'], tempo['a.y'] = zip(*temp)
data=zip(*temp)
data=pd.DataFrame([tempo['a.x'], tempo['a.y']] )
data_trans=data.transpose()
data=data_trans
data.columns=['feature','label']


data=data[~data.index.isin(train['ID'])]

temp.info()
temp.columns=['parsed']
temp.info()
temp['files']=temp.index



temp=temp[~temp['files'].isin(train['ID'])]

#temp.drop('files',axis=1)

train_feat=temp


#parse Test data
temp=test.apply(parserTest,axis=1)
temp.columns=['parsed']
temp['files']=temp.index
temp=temp[~temp['files'].isin(train['ID'])]

test_feat=temp







