from svm.data import load_data_into_classifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
from subjpipeline import Pipeline
from preprocessor import Preprocessor


'''
with open('../vectorized_db.csv', 'a', encoding='utf8') as f:
    f.write('subjective,max swf-isf,avg frs,avg fro,fr frs/fro,frm,pabs,pats\n')
    for i, v in tqdm(enumerate(pipeline.vectors), desc='Writing vectors to disk'):
        f.write(str(pipeline.labels[i]) + ',')
        for j, x in enumerate(v):
            f.write(str(x))
            if j != len(v) - 1:
                f.write(',')
        f.write('\n')
'''

pipeline = Pipeline(Preprocessor('../raw_db.txt')).preprocess()
clf = load_data_into_classifier()
# clf.configure('adam', 'tanh', 0.0001, (3, 4, 5))
clf.fit()
index = -1
sentence = ""
vector = []
for i, s in enumerate(pipeline.sentences):
    if i > 900 and pipeline.vectors[i] in clf.x_test:
        index = i
        sentence = s
        vector = np.array(pipeline.vectors[i]).reshape(1, -1)
        break
print('Test sentence {}: {}'.format(index, sentence))
print('Expected: {}'.format(pipeline.labels[index]))
print('Prediction: {}'.format(clf.classifier.predict(vector)))

# data = pd.read_csv('../vectorized_db.csv', delimiter=',')

# sns.set(style='white', color_codes=True)
# sns.stripplot(x='Subjetiva', y='PATS', data=data, jitter=True, edgecolor='none', alpha=.40)
# sns.pairplot(data, hue='Subjetiva', vars=('MAX SWF-ISF', 'FRM'))
# sns.despine()
# plt.show()

'''
with open('optimal_partition.txt', 'a', encoding='utf8') as f:
    f.write('X TRAIN\n')
    for v in max.x_train:
        f.write(str(v) + '\n')
    f.write('\nY TRAIN\n')
    for v in max.y_train:
        f.write(str(v) + '\n')
    f.write('\nX TEST\n')
    for v in max.x_test:
        f.write(str(v) + '\n')
    f.write('\nY TEST\n')
    for v in max.y_test:
        f.write(str(v) + '\n')
    f.write('\n\n')
    f.write('Kernel: ' + str(max.kernel) + '\n')
    f.write('Precision: ' + str(max.precision) + '\n')
    f.write('Recall: ' + str(max.recall) + '\n')
    f.write('F-Score: ' + str(max.fscore) + '\n')
'''
