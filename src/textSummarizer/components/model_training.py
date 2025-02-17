from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
import torch
from datasets import load_dataset, load_from_disk
from src.textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        config = self.config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        #loading the data
        dataset_samsun_pt = load_from_disk(config.data_path)
        
        trainer_args = TrainingArguments(output_dir = config.root_dir, num_train_epochs= config.num_train_epochs, warmup_steps=config.warmup_steps, 
                                 per_device_train_batch_size = config.per_device_train_batch_size, 
                                 per_device_eval_batch_size = config.per_device_eval_batch_size, weight_decay = config.weight_decay, 
                                 logging_steps = config.logging_steps, eval_strategy = config.eval_strategy, 
                                 eval_steps = config.eval_steps, save_steps = config.save_steps, 
                                 gradient_accumulation_steps = config.gradient_accumulation_steps)

        # Deliberately training on 'test' data as it is smaller size
        trainer = Trainer(model = model, args = trainer_args, processing_class = tokenizer, data_collator=seq2seq_data_collator, 
                          train_dataset=dataset_samsun_pt["test"], eval_dataset=dataset_samsun_pt["validation"])
        
        trainer.train()

        model.save_pretrained(os.path.join(config.root_dir, "pegasus-finetuned-model"))
        tokenizer.save_pretrained(os.path.join(config.root_dir, "tokenizer"))