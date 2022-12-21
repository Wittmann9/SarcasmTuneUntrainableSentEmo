import torch, json, os
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from transformers import AutoModel
from final_classifiers import FinalClassifier, LinearTwoLayersNet
import torch.nn.functional as F
from data_utils import SarcasmDataloader, Word2VecPreprocessing
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from word2vec_arch import CNN_Word2Vec
from transformers import RobertaModel

class TransformerLightning(LightningModule):
    def __init__(self,
                 model_name,
                 emotion_hidden_dim, emotion_middle_dim, emotion_output_dim,
                 sentiment_hidden_dim, sentiment_middle_dim, sentiment_output_dim,
                 transformer_middle_dim, transformer_output_dim,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 w2v_embed_dim=300,
                 labels_output_dim=8,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 milestones=[], gamma=1.0,
                 num_labels=2, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.milestones = milestones
        self.gamma=gamma

        self.model_name = model_name
        self.transformer_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        self.transformer_hidden_transformed = LinearTwoLayersNet(
            input_dim=self.transformer_model.config.hidden_size, middle_dim=transformer_middle_dim, output_dim=transformer_output_dim
        )
        self.emotion_hidden_transformed = LinearTwoLayersNet(
            input_dim=emotion_hidden_dim, middle_dim=emotion_middle_dim, output_dim=emotion_output_dim
        )
        self.sentiment_hidden_transformed = LinearTwoLayersNet(
            input_dim=sentiment_hidden_dim, middle_dim=sentiment_middle_dim, output_dim=sentiment_output_dim
        )

        self.cnn_word2vec = CNN_Word2Vec(
            pretrained_embedding=pretrained_embedding,
            freeze_embedding=freeze_embedding,
            vocab_size=vocab_size,
            embed_dim=w2v_embed_dim,
            filter_sizes=filter_sizes,
            num_filters=num_filters
        )

        self.labels_transform = torch.nn.Sequential(torch.nn.Linear(2+28, labels_output_dim), torch.nn.ReLU())

        self.final_classifier_model = FinalClassifier(
            input_dim=transformer_output_dim + emotion_output_dim + sentiment_output_dim + self.cnn_word2vec.output_dim + labels_output_dim,
            output_dim=num_labels
        )



    def get_transformer_text_representation(self, tokenized_texts):
        model_outputs = self.transformer_model(**tokenized_texts)
        return model_outputs.last_hidden_state[:,0,:]
        return model_outputs.last_hidden_state[:,0,:]/4 + model_outputs.hidden_states[-2][:,0,:]/4 + model_outputs.hidden_states[-3][:,0,:]/4 + model_outputs.hidden_states[-4][:,0,:]/4
 

    def forward(self, batch_dict):

        transformer_tokenized_texts = batch_dict["transformer_tokenized_texts"]
        transformer_representations = self.get_transformer_text_representation(
            tokenized_texts=transformer_tokenized_texts
        )

        transformer_hidden_transformed = self.transformer_hidden_transformed(transformer_representations)
        emotion_hidden_transformed = self.emotion_hidden_transformed(batch_dict['emotion_hidden'])
        sentiment_hidden_transformed = self.sentiment_hidden_transformed(batch_dict['sentiment_hidden'])

        cnn_word2vec_representation = self.cnn_word2vec(batch_dict['word_embedding_indexes'])
        
        labels_representation = self.labels_transform(torch.cat([batch_dict["sentiment_label_distr"],batch_dict["emotion_label_distr"]], dim=-1))

        merged_features = torch.cat(
            [
                transformer_hidden_transformed,
                emotion_hidden_transformed,
                sentiment_hidden_transformed,
                cnn_word2vec_representation,
                labels_representation
               # batch_dict["sentiment_label_distr"],
               # batch_dict["emotion_label_distr"]
            ],
            dim = 1
        )

        final_classifier_logits = self.final_classifier_model(merged_features)
        return final_classifier_logits

    def training_step(self, batch_dict, batch_idx):
        y_hat = self(batch_dict)
        loss = F.cross_entropy(y_hat, batch_dict['labels'])
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
      #  self.trainer.reset_train_dataloader(self)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [ optimizer],[ scheduler]

    def validation_step(self, batch_dict, batch_idx):
        # this is the validation loop
        y_hat = self(batch_dict)
        return {'preds': y_hat, "labels": batch_dict['labels']}

    def validation_epoch_end(self, outputs):
        metrics_dict = self.epoch_end(outputs)

        self.log_dict(metrics_dict, prog_bar=True, logger=True)


    def test_step(self, batch_dict, batch_idx):
        # this is the validation loop
        y_hat = self(batch_dict)
        return {'preds': y_hat, "labels": batch_dict['labels']}

    def test_epoch_end(self, outputs):
        metrics_dict = self.epoch_end(outputs)

        self.log_dict(metrics_dict, prog_bar=True, logger=True)

    def epoch_end(self, outputs):
        predictions = [d['preds'] for d in outputs]
        predictions = torch.cat(predictions)
        pred_classes = torch.argmax(predictions, dim=-1).detach().cpu().numpy()
        labels = [d['labels'] for d in outputs]
        labels = torch.cat(labels).detach().cpu().numpy()

        accuracy = accuracy_score(y_true=labels, y_pred=pred_classes)
        f1 = f1_score(y_true=labels, y_pred=pred_classes)
        recall = recall_score(y_true=labels, y_pred=pred_classes)
        precision = precision_score(y_true=labels, y_pred=pred_classes)

        metrics_dict = {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision
        }
        return metrics_dict



if __name__ == '__main__':
    seed_everything(2)

    TRANSFORMER_PATH = "/home/oxana/SarcasmFeatureExtractor/roberta_tuned"
   # TRANSFORMER_PATH = 'roberta-base'
    TRAIN_DATA_PATH = "/home/oxana/SarcasmFeatureExtractor/v2_merged_without_NC.npy"
    TEST_DATA_PATH = "/home/oxana/SarcasmFeatureExtractor/v2_test_merged_without_NC.npy"
    SAVE_DIR = "v2_emb_roberta_20epoch_bs32_lr5_ml18"
    WORD2VEC_LOADIR = "v2_w2v_embs"
    MAX_LEN = 16
    MAX_EPOCHS = 20
    LR = 1e-5
    MILESTONES = [10]
    GAMMA = 0.1
    BS = 32


    loader = SarcasmDataloader(transformer_model_path=TRANSFORMER_PATH)
    word2vec_processor = Word2VecPreprocessing()
    word2vec_processor.load_from_dir(load_dir=WORD2VEC_LOADIR)
    word2vec_processor.max_len = MAX_LEN

    train_dataloader = loader.get_dataloader(data_file_path=TRAIN_DATA_PATH, batch_size=BS, shuffle=True,
                                             word2vec_processor=word2vec_processor)
    test_dataloader = loader.get_dataloader(data_file_path=TEST_DATA_PATH, batch_size=16, shuffle=False,
                                            word2vec_processor=word2vec_processor)
    # model
    model = TransformerLightning(
        model_name=TRANSFORMER_PATH,
        emotion_hidden_dim=768,
        emotion_middle_dim=264,
        emotion_output_dim=128,
        sentiment_hidden_dim=1024,
        sentiment_middle_dim=264,
        sentiment_output_dim=128,
        transformer_middle_dim=264,
        transformer_output_dim=128,
        pretrained_embedding=torch.Tensor(word2vec_processor.embeddings),
        freeze_embedding=False,
        vocab_size=None,
        w2v_embed_dim=300,
        filter_sizes=[3, 4, 5],
        num_filters=[16, 32, 64],
        num_labels=2, lr=LR,
        milestones=MILESTONES, gamma=GAMMA
    )
    # training
    trainer = Trainer(default_root_dir=SAVE_DIR, accelerator='gpu', gpus=1, max_epochs=MAX_EPOCHS, enable_checkpointing=False)
    
   
   # lr_finder = trainer.tuner.lr_find(model)
   # new_lr = lr_finder.suggestion()

    #print("new_lr: ", new_lr)
   # model.hparams.lr = new_lr
    
    trainer.fit(model, train_dataloader, test_dataloader)

    test_results = trainer.test(
        model=model, dataloaders=test_dataloader
    )

    with open(os.path.join(SAVE_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f)


