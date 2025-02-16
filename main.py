from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_1_data_ingestion import DataIngestionPipeline
from src.textSummarizer.pipeline.stage_2_data_transformation import DataTransformationPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"stage {STAGE_NAME} started")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"stage {STAGE_NAME} finished")

except Exception as e:
    print(e)

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f"stage {STAGE_NAME} started")
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.run_data_transformation()
    logger.info(f"stage {STAGE_NAME} finished")

except Exception as e:
    print(e)

