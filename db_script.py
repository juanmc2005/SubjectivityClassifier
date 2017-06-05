
with open('sentences1495152141.txt', 'r', encoding='utf8') as db:
    class_target = 1000
    subj = 0
    obj = 0

    for line in db.readlines():
        if line.startswith('S'):
            subj += 1
        else:
            obj += 1

    print('Subjetivas: ' + str(subj))
    print('Objetivas: ' + str(obj))
    print('Total: ' + str(subj + obj))
    print('-----------------')
    print('Subjetivas Restantes: ' + str(class_target - subj))
    print('Objetivas Restantes: ' + str(class_target - obj))
    remaining = class_target * 2 - subj - obj
    print('Total Restantes: ' + str(remaining))
    print('-----------------')
    print('Tiempo Estimado: ' + str(remaining / 100) + 'hs')
