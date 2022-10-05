import torch
import json
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification


def chunkify(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]

class FeatureExtractor:
    def __init__(self, model_name, batch_size = 32, device="cpu"):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, output_hidden_states = True)
        self.model.to(device)
        self.device = device

        self.model_name = model_name


    def get_inputs(self, texts):
        tokenized_text = self.tokenizer(
            texts,
            padding = True,
            return_tensors='pt'
        ).to(self.device)
        return tokenized_text

    def get_logits(self, model_output):
        logits = model_output.logits.cpu()
        logits = torch.softmax(logits, dim=-1)
        return logits.numpy()

    def get_last_hidden(self, model_output):
        return model_output.hidden_states[-1].cpu().numpy()

    def save_features(self, logits, last_hiddens, batch_dicts):
        for logit, last_hidden, input_dict in zip(logits, last_hiddens, batch_dicts):
            new_features_dict = {
                    'label_distr': logit,
                    'last_hidden': last_hidden,
                }
            input_dict.update(new_features_dict)

    def get_texts(self, list_of_dicts):
        return [element['text'] for element in list_of_dicts]

    def predict_features(self, dicts_list):
        with torch.no_grad():
            for batch_dicts in chunkify(dicts_list, self.batch_size):
                batch_text = self.get_texts(batch_dicts)
                model_input = self.get_inputs(batch_text)
                model_output = self.model(**model_input)
                logits = self.get_logits(model_output)
                last_hidden = self.get_last_hidden(model_output)

                self.save_features(logits, last_hidden, batch_dicts)

        return dicts_list

    def save_nc_features(self, logits, last_hidden, nc_dict):
        nc_dict[self.model_name+"_nc_features"] = {
                    'label_distr': logits,
                    'last_hidden': last_hidden,
                }

    def predict_features_from_NC(self, dicts_list):
        with torch.no_grad():
            for nc_dict in dicts_list:
                noun_chunks = nc_dict['noun_chunks']
                if len(noun_chunks) == 0:
                    noun_chunks = ["_ nothing _"]
                    nc_dict["noun_chunks"] = noun_chunks
                model_input = self.get_inputs(noun_chunks)
                model_output = self.model(**model_input)
                logits = self.get_logits(model_output)
                last_hidden = self.get_last_hidden(model_output)

                self.save_nc_features(logits, last_hidden, nc_dict)

        return dicts_list





if __name__ == '__main__':

    with open('reddit_train.json', 'r') as openfile:
        #     Reading from json file
        input_dicts = json.load(openfile)
    fs = FeatureExtractor(model_name='bhadresh-savani/bert-base-go-emotion', batch_size=16)
    features = fs.predict_features_from_NC(input_dicts)
    # print(features)

    np.save('reddit_fetures_emotion', features)