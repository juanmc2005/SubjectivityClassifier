from preprocessor import Preprocessor
from subjclassifier import Pipeline


print(Pipeline(Preprocessor('../raw_db.txt')).preprocess().optimal_svm())

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
