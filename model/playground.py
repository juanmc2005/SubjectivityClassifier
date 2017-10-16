from nn_cross.data import load_data_into_classifier as load_nn
from svm_cross.data import load_data_into_classifier as load_svm
import numpy as np
from subjpipeline import Pipeline
from preprocessor import Preprocessor
from sklearn.metrics import precision_recall_fscore_support as scores
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


pipeline = Pipeline(Preprocessor('../raw_db.txt'))\
    .preprocess()\
    .load_svm()

vectors, predictions, estimated_labels = pipeline.predict('../prod_db.txt')
precision, recall, fscore, support = scores(pipeline.labels, predictions)
success = 0
for label, prediction in zip(pipeline.labels, predictions):
    if label == prediction:
        success += 1
success_rate = success / len(predictions)
print("""
    Precision: {}
    Recall: {}
    F-Score: {}
    Success Rate: {}
""".format(precision, recall, fscore, success_rate))


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

# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=30, random_state=28756285)

'''
nn = load_nn()
nn.configure('adam', 'relu', 0.01, (3,))
nn.fit()
nn_precision, nn_recall, nn_fscore = nn.evaluate()

f1meannn = cross_val_score(nn.classifier, pipeline.vectors, pipeline.labels, cv=cv, scoring='recall').mean()
print("""
    NN K Fold Mean Recall: {}
""".format(f1meannn))
'''




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
