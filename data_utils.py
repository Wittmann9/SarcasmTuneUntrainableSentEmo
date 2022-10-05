from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from transformers import AutoTokenizer
import torch


class SarcasmDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.load(file_path, allow_pickle = True)

    def __getitem__(self, item):
        text = self.data[item]['text']
        emotion_text_vector = self.data[item]['emotion']['last_hidden'][0]
        sentiment_text_vector = self.data[item]['sentiment']['last_hidden'][0]
        emotion_label_distr = self.data[item]['emotion']['label_distr']
        sentiment_label_distr = self.data[item]['sentiment']['label_distr']
        label = self.data[item]['label']

        return {
            'text': text,
            'emotion_text_vector': emotion_text_vector,
            'sentiment_text_vector': sentiment_text_vector,
            'emotion_label_distr': emotion_label_distr,
            'sentiment_label_distr': sentiment_label_distr,
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
        labels = torch.Tensor([d['label'] for d in batch_list]).long()

        return {
            "transformer_tokenized_texts": transformer_tokenized_texts,
            "emotion_hidden": emotion_hidden,
            "sentiment_hidden": sentiment_hidden,
            "labels": labels
        }

    def get_dataloader(self,
            data_file_path, batch_size, shuffle,
            drop_last = False):
        dataset = SarcasmDataset(data_file_path)
        loader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=self.collate_fn)
        return loader

