import os
import re
from watson_developer_cloud import NaturalLanguageUnderstandingV1
import watson_developer_cloud.natural_language_understanding.features.v1 as Features
from tqdm import tqdm

FILE = 'continueAprendizDeConspirador.txt'

username = os.environ['BM_USER']
password = os.environ['BM_PASS']

nlu = NaturalLanguageUnderstandingV1(version='2017-02-27', username=username, password=password)

scount = 0
ocount = 0
with open(FILE, 'r', encoding='utf8') as f,\
        open('obj.txt', 'a', encoding='utf8') as obj,\
        open('subj.txt', 'a', encoding='utf8') as subj:
    lines = f.read().replace('\n', ' ').replace(':', '. ').replace('--', '. ').split('. ')
    lines = list(filter(lambda x: len(re.sub(' +', ' ', x.strip()).split(' ')) > 8, lines))
    for sentence in tqdm(lines[200:600], desc='Analyzing sentences'):
        s = re.sub(' +', ' ', sentence.strip())
        response = nlu.analyze(text=s, features=[Features.Sentiment()], language='es')
        score = response['sentiment']['document']['score']
        if score >= 0.5 or score <= -0.5:
            subj.write(s + '\n')
            scount += 1
        else:
            obj.write(s + '\n')
            ocount += 1

print('Objective Count: ' + str(ocount))
print('Subjective Count: ' + str(scount))
