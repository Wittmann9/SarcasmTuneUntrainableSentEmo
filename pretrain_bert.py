from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)



model_args = LanguageModelingArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 1
model_args.dataset_type = "simple"
model_args.evaluate_during_training_steps = -1
model_args.learning_rate = 4e-5
model_args.manual_seeds = 42
model_args.max_seq_length = 128
model_args.save_model_every_epoch = False
model_args.save_optimizer_and_scheduler = False
model_args.save_steps = -1
model_args.train_batch_size = 8
model_args.output_dir = 'distilbert_tuned/'

train_file = "all_trains_merged.txt"
# test_file = "data/test.txt"

model = LanguageModelingModel(
    "distilbert", "distilbert-base-uncased", args=model_args
)

# Train the model
model.train_model(train_file)