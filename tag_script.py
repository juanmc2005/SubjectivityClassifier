'''
This was the script used to filter only spanish papers

from shutil import copyfile

with open('texts/Metadata_es.txt', 'r', encoding='ISO-8859-1') as md:
    for line in md.readlines():
        name = line.split(';')[0]
        copyfile('texts/' + name, 'papers/' + name)
'''


import time


PATH = 'continue1495307914.txt'
TIMESTAMP = str(round(time.time()))


def open_out_file():
    return open('sentences' + TIMESTAMP + '.txt', 'a', encoding='utf8')


def open_remaining_file():
    return open('continue' + TIMESTAMP + '.txt', 'a', encoding='utf8')


def open_paper(path):
    return open(path, 'r', encoding='utf8')


def ask_label(sentence, current_count, out):
    count = 0
    if sentence != '' and len(sentence.split(' ')) <= 50:
        print(sentence)
        res = 'a'
        while res not in 'sod':
            print('Subjetiva (S) - Objetiva (O) - Descartar (D) - Salir (X) ---- CLASIFICADAS: ' + str(current_count))
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
    with open_paper(path) as paper, open_out_file() as out, open_remaining_file() as remainings:
        text = paper.read().replace('\n', ' ')
        ok = True
        for sentence in text.split('. '):
            for line in sentence.split('-'):
                if ok:
                    ok, res = ask_label(line.strip(), count, out)
                    count += res
                    if not ok:
                        remainings.write(line + '.\n')
                else:
                    remainings.write(line + '.\n')


append_paper_labels(PATH)
