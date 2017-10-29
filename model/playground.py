from nn_cross.data import load_data_into_classifier as load_nn
from svm_cross.data import load_data_into_classifier as load_svm
import numpy as np
from subjpipeline import Pipeline
from preprocessor import Preprocessor
from sklearn.metrics import precision_recall_fscore_support as scores
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


# pipeline = Pipeline(Preprocessor('../raw_db.txt'))\
#     .preprocess()\
#     .optimal_nn()



'''
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

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=30, random_state=28756285)

nn = load_nn()
nn.configure('adam', 'relu', 0.01, (3,))
nn.fit()
nn_precision, nn_recall, nn_fscore = nn.evaluate()

vals = cross_val_score(nn.classifier, pipeline.vectors, pipeline.labels, cv=cv, scoring='f1_macro')

print("""
    Scores: {}
    SVM K Fold Mean F-Score: {}
    SVM K Fold Standard Deviation F-Score: {}
""".format(vals, vals.mean(), vals.std()))

'''
vals = [0.87949085,  0.89950759,  0.92241805,  0.88203922,  0.92468178,
          0.90205373,  0.90227952,  0.9222858,   0.90205373,  0.87956644,
          0.88690555,  0.90733262,  0.90713725,  0.89943432,  0.90480723,
          0.8921434,   0.8996387,   0.8972167,   0.91717647,  0.90211765,
          0.89726821,  0.88682021,  0.90724434,  0.89950759,  0.9197995,
          0.87685271,  0.89703085,  0.94239595,  0.88705882,  0.9047619,
          0.91220974,  0.87932118,  0.91234167,  0.89974937,  0.90198481,
          0.89448297,  0.89974937,  0.8813723,   0.91469289,  0.91747473,
          0.92721569,  0.88729643,  0.89207843,  0.88937775,  0.91225816,
          0.872,       0.90480723,  0.91717647,  0.9044626,   0.90724434,
          0.91458145,  0.86926461,  0.90713725,  0.90981738,  0.90733262,
          0.89950759,  0.90967483,  0.90227952,  0.89473684,  0.90205373,
          0.90955683,  0.91727198,  0.89220288,  0.89200796,  0.8972167,
          0.91717647,  0.88203922,  0.90459692,  0.91983767,  0.8845845,
          0.89478694,  0.93492028,  0.90719316,  0.90205373,  0.86881606,
          0.88187913,  0.92224292,  0.91469289,  0.8921434,   0.89726821,
          0.91731344,  0.89473684,  0.87666052,  0.91727198,  0.90211765,
          0.91975727,  0.88945835,  0.88705882,  0.90988961,  0.89943432,
          0.8972167,   0.89703085,  0.90955683,  0.91983767,  0.88705882,
          0.90227952,  0.90205373,  0.90724434,  0.89715999,  0.89950759,
          0.89957571,  0.88966624,  0.90724434,  0.8916706,   0.91738382,
          0.88451206,  0.90977444,  0.90205373,  0.90211765,  0.90724434,
          0.89943432,  0.88471178,  0.92484782,  0.89455449,  0.90713725,
          0.90967483,  0.90223052,  0.89423585,  0.88972431,  0.91230213,
          0.89193197,  0.90724434,  0.91727198,  0.89200796,  0.90223052,
          0.8921434,   0.91478697,  0.88187913,  0.91474209,  0.90713725,
          0.88705882,  0.89695839,  0.90972692,  0.89709804,  0.91735071,
          0.90713725,  0.90198481,  0.89478694,  0.88945835,  0.91482753,
          0.88929147,  0.90223052,  0.90724434,  0.88960257,  0.9197995,
          0.90211765,  0.89423585,  0.89478694,  0.91225816,  0.90471175]
'''

from pprint import pprint

mat = np.reshape(vals, (30, 5))
res = []
for arr in mat:
    res.append(arr.mean())
res = np.array(res)

pprint(res)

print("""
    Mean: {}
    Standard Deviation: {}
""".format(res.mean(), res.std()))



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
