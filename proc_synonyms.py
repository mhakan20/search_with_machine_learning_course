import fasttext

syn_model = fasttext.load_model('/workspace/datasets/fasttext/title_model.bin')
synonyms = open("/workspace/datasets/fasttext/synonyms.csv", 'w')

tr = 0.5

for w in open('/workspace/datasets/fasttext/top_words.txt', 'r').readlines():
    results = syn_model.get_nearest_neighbors(w.replace('\n', ''))
    close_syns = [val[1] for val in results if val[0] >= tr]
    if len(close_syns) > 0:
        new_line = w.replace('\n', '')
        for syn in close_syns:
            new_line = new_line + ',' + syn
        new_line = new_line + '\n'
        synonyms.write(new_line)
