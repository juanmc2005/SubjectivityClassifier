'''
This was the script used to filter only spanish papers

from shutil import copyfile

with open('texts/Metadata_es.txt', 'r', encoding='ISO-8859-1') as md:
    for line in md.readlines():
        name = line.split(';')[0]
        copyfile('texts/' + name, 'papers/' + name)
'''


import time
import re


# PATH = 'res/books/pg47103.txt'
PATH = 'continue1496622301.txt'
TIMESTAMP = str(round(time.time()))


def open_out_file():
    return open('sentences' + TIMESTAMP + '.txt', 'a', encoding='utf8')


def open_remaining_file():
    return open('continue' + TIMESTAMP + '.txt', 'a', encoding='utf8')


def open_paper(path):
    return open(path, 'r', encoding='utf8')


def ask_label(sentence, current_count, remaining, out):
    count = 0
    if sentence != '' and len(sentence.split(' ')) <= 50:
        print(sentence)
        res = 'a'
        while res not in 'sod':
            print('Subjetiva (S) - Objetiva (O) - Descartar (D) - Salir (X) ---- CLASIFICADAS: ' + str(current_count) + ' ---- RESTANTES: ' + str(remaining))
            res = input('').lower()
            if res == 's':
                count += 1
                out.write('S@' + sentence + '\n')
                print('Subjetiva\n')
            elif res == 'o':
                count += 1
                out.write('O@' + sentence + '\n')
                print('Objetiva\n')
            elif res == 'd':
                print('Descartada\n')
            elif res == 'x':
                return False, 0
            else:
                print('Caracter Inválido\n')
    return True, count


def append_paper_labels(path):
    count = 0
    asked = 0
    with open_paper(path) as paper, open_out_file() as out, open_remaining_file() as remainings:
        lines = paper.read().replace('\n', ' ').replace(':', '. ').split('. ')
        total = len(lines)
        ok = True
        for sentence in lines:
            if ok:
                ok, res = ask_label(re.sub(' +', ' ', sentence.strip()), count, total - asked, out)
                count += res
                asked += 1
                if not ok:
                    remainings.write(sentence + '.\n')
            else:
                remainings.write(sentence + '.\n')


append_paper_labels(PATH)
