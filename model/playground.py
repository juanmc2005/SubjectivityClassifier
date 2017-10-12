from nn_cross.data import load_data_into_classifier as load_nn
from svm_cross.data import load_data_into_classifier as load_svm
import numpy as np
from subjpipeline import Pipeline
from preprocessor import Preprocessor
from sklearn.metrics import precision_recall_fscore_support as scores


pipeline = Pipeline(Preprocessor('../raw_db.txt'))
pipeline.preprocess()


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

nn = load_nn()
nn.configure('adam', 'relu', 0.01, (3,))
nn.fit()
nn_precision, nn_recall, nn_fscore = nn.evaluate()

svm = load_svm()
svm.configure('sigmoid', 0.01, 0.001)
svm.fit()
svm_precision, svm_recall, svm_fscore = svm.evaluate()

predicted = []
for v in pipeline.vectors:
    y_nn = nn.classifier.predict(np.reshape(v, (1, -1)))
    y_svm = svm.classifier.predict(np.reshape(v, (1, -1)))
    if y_nn == 1 and y_svm == 1:
        predicted.append(1)
    elif y_nn == 0 and y_svm == 0:
        predicted.append(0)
    elif nn_precision[y_nn] > svm_precision[y_svm]:
        predicted.append(y_nn)
    else:
        predicted.append(y_svm)

precision, recall, fscore, support = scores(pipeline.labels, predicted)
print("""
    precision: {},
    recall: {},
    fscore: {}
""".format(precision, recall, fscore))



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
