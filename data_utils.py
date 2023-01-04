from torch.utils.data import Dataset, DataLoader
import json, os
import numpy as np
from transformers import AutoTokenizer
import torch
from nltk.tokenize import word_tokenize
from tqdm import tqdm
np.random.seed(2)


class Word2VecPreprocessing:
    def __init__(self, word2idx = None, embeddings = (), max_len=10):
        self.word2idx = {}
        if word2idx is not None:
            self.word2idx = word2idx

        self.embeddings = None
        if len(embeddings):
            self.embeddings = embeddings

        self.max_len = max_len
        self.w2v_voc = []

    def tokenize(self, text):
        return word_tokenize(text)

    def build_word2idx(self, tokenized_texts):
        """ word2idx (Dict): Vocabulary built from the corpus """

        # add <pad> and <unk> tokens to the vocabulary
        self.word2idx['<pad>'] = 0
        self.word2idx['<unk>'] = 1

        # Building our vocab from the corpus starting from index 2
        idx = 2
        for tokenized_sent in tokenized_texts:
            for token in tokenized_sent:
                if token not in self.word2idx:
                    self.word2idx[token] = idx
                    idx += 1


    def load_pretrained_vectors(self, fname):
        """Load pretrained vectors and create embedding layers.

        Args:
            word2idx (Dict): Vocabulary built from the corpus
            fname (str): Path to pretrained vector file

        Returns:
            embeddings (np.array): Embedding matrix with shape (N, d) where N is
                the size of word2idx and d is embedding dimension
        """

        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        d = len(fin.readline().split(" ")) - 1
        # n, d = map(int, fin.readline().split())

        # Initilize random embeddings
        embeddings = np.random.uniform(-0.25, 0.25, (len(self.word2idx), d))
        embeddings[self.word2idx['<pad>']] = np.zeros((d,))

        # Load pretrained vectors
        count = 0
        for line in tqdm(fin):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in self.word2idx:
                count += 1
                embeddings[self.word2idx[word]] = np.array(tokens[1:], dtype=np.float32)
        return embeddings


    def get_average_w2v_embeddings(self, text):
        tokens = self.tokenize(text)
        tokens = [token for token in tokens if token in self.w2v_voc]
        if len(tokens) == 0:
            return self.embeddings[self.word2idx['<pad>']]

        tokens_embeddings = [self.embeddings[self.word2idx[word]] for word in tokens]

        return np.average(tokens_embeddings, axis = 0)


    def load_w2v_vocabulary(self, fname):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        for line in tqdm(fin):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in self.word2idx:
                self.w2v_voc.append(word)


    def save_embeddings_word2idx(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "embeddig_matrix"), self.embeddings)

        with open(os.path.join(save_dir, "word2idx.json"), "w") as f:
            json.dump(self.word2idx, f)

    def load_from_dir(self, load_dir):
        self.embeddings = np.load(os.path.join(load_dir, "embeddig_matrix.npy"))
        with open(os.path.join(load_dir, "word2idx.json"), "r") as f:
            self.word2idx = json.load(f)

    def get_weights_word2idx(self, texts, emb_file, save_dir=None):

        tokenized_texts = [self.tokenize(text) for text in texts]
        self.build_word2idx(tokenized_texts=tokenized_texts)

        self.embeddings = self.load_pretrained_vectors(fname=emb_file)

        if save_dir:
            self.save_embeddings_word2idx(save_dir)

    def encode(self, text):
        tokens = self.tokenize(text)

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += ['<pad>'] * (self.max_len - len(tokens))

        input_id = [self.word2idx.get(token, 1) for token in tokens]
        return input_id


class SarcasmDataset(Dataset):
    def __init__(self, file_path, word2vec_processor):
        self.data = np.load(file_path, allow_pickle = True)
        self.word2vec_processor = word2vec_processor

    def __getitem__(self, item):
        text = self.data[item]['text']
   #     average_w2v_embeddings = self.word2vec_processor.get_average_w2v_embeddings(text)
        emotion_text_vector = self.data[item]['emotion']['last_hidden'][0]
        sentiment_text_vector = self.data[item]['sentiment']['last_hidden'][0]
        emotion_label_distr = self.data[item]['emotion']['label_distr']
        sentiment_label_distr = self.data[item]['sentiment']['label_distr']
        word_embedding_indexes = self.word2vec_processor.encode(self.data[item]['text'])
        label = self.data[item]['label']

        return {
            'text': text,
  #          'average_w2v_embeddings': average_w2v_embeddings,
            'emotion_text_vector': emotion_text_vector,
            'sentiment_text_vector': sentiment_text_vector,
            'emotion_label_distr': emotion_label_distr,
            'sentiment_label_distr': sentiment_label_distr,
            'word_embedding_indexes': word_embedding_indexes,
            'label': label
        }
    def __len__(self):
        return len(self.data)


class SarcasmDataloader:
    def __init__(self, transformer_model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)

    def collate_fn(self, batch_list):
        transformer_tokenized_texts = self.tokenizer(
            [d['text'] for d in batch_list],
            padding=True,
            return_tensors='pt',
            truncation=True
        )
        emotion_hidden = torch.Tensor([d['emotion_text_vector'] for d in batch_list])
        sentiment_hidden = torch.Tensor([d['sentiment_text_vector'] for d in batch_list])
        emotion_label_distr = torch.Tensor([d['emotion_label_distr'] for d in batch_list])
        sentiment_label_distr = torch.Tensor([d['sentiment_label_distr'] for d in batch_list])
        labels = torch.Tensor([d['label'] for d in batch_list]).long()

        word_embedding_indexes = torch.Tensor([d["word_embedding_indexes"] for d in batch_list]).long()
 #       average_w2v_embeddings = torch.Tensor([d['average_w2v_embeddings'] for d in batch_list])
        return {
#            'average_w2v_embeddings': average_w2v_embeddings,
            "transformer_tokenized_texts": transformer_tokenized_texts,
            "emotion_hidden": emotion_hidden,
            "sentiment_hidden": sentiment_hidden,
            "word_embedding_indexes": word_embedding_indexes,
            "labels": labels,
            "emotion_label_distr": emotion_label_distr,
            "sentiment_label_distr": sentiment_label_distr
        }

    def get_dataloader(self,
            data_file_path, batch_size, shuffle, word2vec_processor,
            drop_last = False):

        dataset = SarcasmDataset(data_file_path, word2vec_processor)
        loader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=self.collate_fn)
        return loader

