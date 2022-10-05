import torch, json, os
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer
from final_classifiers import FinalClassifier, LinearTwoLayersNet
import torch.nn.functional as F
from data_utils import SarcasmDataloader
from data_utils import SarcasmDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

class TransformerLightning(LightningModule):
    def __init__(self,
                 model_name,
                 emotion_hidden_dim, emotion_middle_dim, emotion_output_dim,
                 sentiment_hidden_dim, sentiment_middle_dim, sentiment_output_dim,
                 transformer_middle_dim, transformer_output_dim,
                 num_labels=2, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model_name = model_name
        self.transformer_model = AutoModel.from_pretrained(model_name)
        self.final_classifier_model = FinalClassifier(
            input_dim=transformer_output_dim + emotion_output_dim + sentiment_output_dim,
            output_dim=num_labels
        )

        self.transformer_hidden_transformed = LinearTwoLayersNet(
            input_dim=self.transformer_model.config.dim, middle_dim=transformer_middle_dim, output_dim=transformer_output_dim
        )
        self.emotion_hidden_transformed = LinearTwoLayersNet(
            input_dim=emotion_hidden_dim, middle_dim=emotion_middle_dim, output_dim=emotion_output_dim
        )
        self.sentiment_hidden_transformed = LinearTwoLayersNet(
            input_dim=sentiment_hidden_dim, middle_dim=sentiment_middle_dim, output_dim=sentiment_output_dim
        )



    def get_transformer_text_representation(self, tokenized_texts):
        model_outputs = self.transformer_model(**tokenized_texts)
        return model_outputs.last_hidden_state[:,0,:]


    def forward(self, batch_dict):

        transformer_tokenized_texts = batch_dict["transformer_tokenized_texts"]
        transformer_representations = self.get_transformer_text_representation(
            tokenized_texts=transformer_tokenized_texts
        )

        # merge these vectors
        transformer_hidden_transformed = self.transformer_hidden_transformed(transformer_representations)
        emotion_hidden_transformed = self.emotion_hidden_transformed(batch_dict['emotion_hidden'])
        sentiment_hidden_transformed = self.sentiment_hidden_transformed(batch_dict['sentiment_hidden'])

        merged_features = torch.cat(
            [
                transformer_hidden_transformed,
                emotion_hidden_transformed,
                sentiment_hidden_transformed
            ],
            dim = 1
        )

        final_classifier_logits = self.final_classifier_model(merged_features)
        return final_classifier_logits

    def training_step(self, batch_dict, batch_idx):
        y_hat = self(batch_dict)
        loss = F.cross_entropy(y_hat, batch_dict['labels'])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

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

    TRANSFORMER_PATH = ""
    TRAIN_DATA_PATH = ""
    TEST_DATA_PATH = ""
    SAVE_DIR = ""

    loader = SarcasmDataloader(transformer_model_path=TRANSFORMER_PATH)
    train_dataloader = loader.get_dataloader(data_file_path=TRAIN_DATA_PATH, batch_size=32, shuffle=True)
    test_dataloader = loader.get_dataloader(data_file_path=TEST_DATA_PATH, batch_size=32, shuffle=False)
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
        num_labels=2, lr=1e-3
    )
    # training
    trainer = Trainer(default_root_dir=SAVE_DIR, accelerator='gpu', gpus=1, enable_checkpointing=False)
    trainer.fit(model, train_dataloader, test_dataloader)

    test_results = trainer.test(
        model=model, dataloaders=test_dataloader
    )

    with open(os.path.join(SAVE_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f)


