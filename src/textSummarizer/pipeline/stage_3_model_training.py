from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_training import ModelTrainer

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def run_model_training(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(model_trainer_config)
        model_trainer.train()