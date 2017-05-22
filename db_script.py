
with open('sentences1495152141.txt', 'r', encoding='utf8') as db:
    subj = 0
    obj = 0

    for line in db.readlines():
        if line.startswith('S'):
            subj += 1
        else:
            obj += 1

    print('Subjetivas: ' + str(subj))
    print('Objetivas: ' + str(obj))