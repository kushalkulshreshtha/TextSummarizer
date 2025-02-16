import os
from src.textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from src.textSummarizer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_to_features(self, batch):
    # print(batch)
        input_tokens = self.tokenizer(batch['dialogue'], max_length=1024, truncation=True)
        output_tokens = self.tokenizer(batch['summary'], max_length=128, truncation=True)
    
        return {
            'input_ids' : input_tokens['input_ids'],
            'attention_mask': input_tokens['attention_mask'],
            'labels': output_tokens['input_ids']
        }

    def convert(self):
        dataset_samsun = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsun.map(self.convert_to_features, batched = True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, 'samsun_dataset'))