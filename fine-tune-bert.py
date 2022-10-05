# !pip3 install transformers
#
# !pip3 install simpletransformers

from simpletransformers.language_modeling import LanguageModelingModel

model_args = {
    "num_train_epochs": 5,
    'evaluate_during_training_steps' : False,
    'train_batch_size' : 8,
    'ave_eval_checkpoints': False,
    'save_optimizer_and_scheduler': False,
    'save_steps': -1
}

model = LanguageModelingModel(
    "bert", "bert-base-uncased",
    use_cuda=True,
    args = model_args
)
model.train_model("reddit_v2_texts.txt")



