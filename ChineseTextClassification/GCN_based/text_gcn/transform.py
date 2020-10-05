import jieba

stopwords = [x.strip()
             for x in open('.\data\stopwords.txt', encoding='utf8').readlines()]

vocab_dic = {}
doc_content_list = []
label = []
trainortest = []


def tokenizer(x):
    return (y for y in x)


def get_clean_text(text):
    text = jieba.cut(text)
    text = [word for word in text if word not in stopwords]
    return ''.join(text)


def load_data(path, train=True):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            content, _label = line.strip().split('\t')
            if train:
                trainortest.append('train')
            else:
                trainortest.append('test')
            content = get_clean_text(content)
            doc_content_list.append(content)
            label.append(_label)
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1


load_data("data/train.txt")
load_data("data/test.txt", train=False)

clean_docs, clean_label = [], []
threashold = 0
print(f"words count: {len(vocab_dic)}")
print(
    f"words used: {len([cnt for _, cnt in vocab_dic.items() if cnt >= threashold])}")

for doc_content in doc_content_list:
    doc_words = []
    for word in tokenizer(doc_content):
        if word != ' ' and vocab_dic[word] >= threashold:
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)

for i in range(len(label)):
    tmp = str(i) + '\t' + trainortest[i] + '\t' + label[i]
    clean_label.append(tmp)


clean_corpus_str = '\n'.join(clean_docs)
with open('data/corpus/r8.clean.txt', 'w', encoding='utf-8') as f:
    f.write(clean_corpus_str)

clean_str = '\n'.join(clean_label)
with open('data/r8.txt', 'w', encoding='utf-8') as f:
    f.write(clean_str)
