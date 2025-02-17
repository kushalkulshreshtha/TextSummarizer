from src.textSummarizer.constants import CONFIG_FILEPATH, PARAMS_FILEPATH
from src.textSummarizer.utils.common import read_yaml, create_directories
from src.textSummarizer.entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig

class ConfigurationManager:
    def __init__(self, config_path = CONFIG_FILEPATH, params_path = PARAMS_FILEPATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(root_dir = config.root_dir, 
                                                   source_URL = config.source_URL,
                                                   local_data_file = config.local_data_file,
                                                   unzip_dir = config.unzip_dir)
        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        data_tranformation_config = DataTransformationConfig(root_dir= config.root_dir,
                                                            data_path=config.data_path,
                                                            tokenizer_name=config.tokenizer_name)
        return data_tranformation_config
    
    def get_model_trainer_config(self):
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        create_directories([config.root_dir])
        
        mtc = ModelTrainerConfig(root_dir = config.root_dir, data_path=config.data_path, model_ckpt = config.model_ckpt,
                                num_train_epochs=params.num_train_epochs, warmup_steps = params.warmup_steps, 
                                 per_device_train_batch_size = params.per_device_train_batch_size,  
                                 per_device_eval_batch_size = params.per_device_eval_batch_size,
                                 weight_decay = params.weight_decay, logging_steps = params.logging_steps, eval_strategy = params.eval_strategy,
                                 eval_steps = params.eval_steps, save_steps = float(params.save_steps), 
                                 gradient_accumulation_steps = params.gradient_accumulation_steps)
        return mtc