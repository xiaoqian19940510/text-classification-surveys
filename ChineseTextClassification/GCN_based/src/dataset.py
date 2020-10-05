import os
import pickle

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer

UNK, PAD = '[UNK]', '[PAD]'  # 未知，padding符号
train_path = "../data/train.txt"
dev_path = "../data/dev.txt"
test_path = "../data/test.txt"
word2vec_path = "../data/sgns.sogou.char"
trimmed_path = "../data/embedding.npz"
vocab_path = "../data/vocab.pkl"
vocab_size = 122777  # a prime number for hash
class_list = [x.strip()
              for x in open('..\data\class.txt', encoding='utf8').readlines()]

bert_path = "../data/bert"
CLS = '[CLS]'  # for bert classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenizer(x):
    return [y for y in x]


def build_vocab(file_path):
    vocab_dic = {}  # word counts
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in f:
            content = line.strip().split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        print(f"words tot: {len(vocab_dic)}")
        vocab_list = sorted([_ for _ in vocab_dic.items()], key=lambda x: x[1], reverse=True)
        vocab_dic = {word_count[0]: idx for idx,
                     word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def load_word2id():
    if os.path.exists(vocab_path):
        word2id = pickle.load(open(vocab_path, 'rb'))
    else:
        word2id = build_vocab(train_path)
        pickle.dump(word2id, open(vocab_path, 'wb'))
    return word2id


def load_data(input_path, pad_size):
    word2id = load_word2id()

    tokens, labels = [], []
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            text, label = line.strip().split('\t')
            labels.append(int(label))

            token_ids = []
            for token in tokenizer(text)[:pad_size]:
                token_ids.append(word2id.get(token, word2id[UNK]))
            if len(token) < pad_size:  # padding
                token_ids.extend([word2id[PAD]] * (pad_size - len(token_ids)))
            tokens.append(token_ids)

    tokens = torch.LongTensor(tokens).to(device)
    labels = torch.LongTensor(labels).to(device)
    return tokens, labels


def load_data_for_fastText(input_path, pad_size):
    word2id = load_word2id()

    def bigram_hash(sequence, index):
        t1 = sequence[index - 1] if index - 1 >= 0 else 0
        return (t1 * 16341163) % vocab_size

    def trigram_hash(sequence, index):
        t1 = sequence[index - 1] if index - 1 >= 0 else 0
        t2 = sequence[index - 2] if index - 2 >= 0 else 0
        return (t2 * 16341163 * 12255871 + t1 * 16341163) % vocab_size

    tokens, tokens_bigram, tokens_trigram, labels = [], [], [], []
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            text, label = line.strip().split('\t')
            labels.append(int(label))

            token_ids = []
            for token in tokenizer(text)[:pad_size]:
                token_ids.append(word2id.get(token, word2id[UNK]))
            if len(token) < pad_size:  # padding
                token_ids.extend([word2id[PAD]] * (pad_size - len(token_ids)))
            # get ngram     
            token_ids_bigram = [bigram_hash(token_ids, index) for index in range(pad_size)]
            token_ids_trigram = [trigram_hash(token_ids, index) for index in range(pad_size)]

            tokens.append(token_ids)
            tokens_bigram.append(token_ids_bigram)
            tokens_trigram.append(token_ids_trigram)

    tokens = torch.LongTensor(tokens).to(device)
    tokens_bigram = torch.LongTensor(tokens_bigram).to(device)
    tokens_trigram = torch.LongTensor(tokens_trigram).to(device) 
    labels = torch.LongTensor(labels).to(device)
    return tokens, tokens_bigram, tokens_trigram, labels        
    

def load_data_for_BERT(input_path, pad_size):
    tokens, masks, labels = [], [], []
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            text, label = line.strip().split('\t')
            labels.append(int(label))

            token = tokenizer.tokenize(text)
            token = [CLS] + token
            mask = []
            token_ids = tokenizer.convert_tokens_to_ids(token)

            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
            tokens.append(token_ids)
            masks.append(mask)
    
    tokens = torch.LongTensor(tokens).to(device)
    masks = torch.LongTensor(masks).to(device)
    labels = torch.LongTensor(labels).to(device)
    return tokens, masks, labels


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, padding_len, mode=None):
        super().__init__()
        self.mode = mode
        if mode == "bert":
            self.tokens, self.masks, self.labels = load_data_for_BERT(input_path, padding_len)
        elif mode == 'fastText':
            self.tokens, self.tokens_bigram, self.tokens_trigram, self.labels = load_data_for_fastText(input_path, padding_len)
        else:
            self.tokens, self.labels = load_data(input_path, padding_len)
        print(f"loaded {input_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.mode == 'bert':
            return (self.tokens[index], self.masks[index]), self.labels[index]
        if self.mode == 'fastText':
            return (self.tokens[index], self.tokens_bigram[index], self.tokens_trigram[index]), self.labels[index]
        return self.tokens[index], self.labels[index]


def main():
    emb_dim = 300
    word2id = load_word2id()

    embeddings = np.random.rand(len(word2id), emb_dim)
    with open(word2vec_path, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip().split(' ')
            if line[0] in word2id:
                idx = word2id[line[0]]
                emb = [float(x) for x in line[1:]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
    np.savez_compressed(trimmed_path, embeddings=embeddings)


if __name__ == "__main__":
    main()
