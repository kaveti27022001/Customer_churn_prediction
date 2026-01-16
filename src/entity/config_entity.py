import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Lazy initialization - will be set on first access
_training_pipeline_config = None

def get_training_pipeline_config():
    """Factory function to create TrainingPipelineConfig with lazy imports"""
    global _training_pipeline_config
    if _training_pipeline_config is None:
        from src.constants import PIPELINE_NAME, ARTIFACT_DIR
        
        class TrainingPipelineConfig:
            def __init__(self):
                self.pipeline_name = PIPELINE_NAME
                self.artifact_dir = os.path.join(ARTIFACT_DIR, TIMESTAMP)
                self.timestamp = TIMESTAMP
        
        _training_pipeline_config = TrainingPipelineConfig()
    
    return _training_pipeline_config


@property 
def training_pipeline_config():
    """Property to lazily get training_pipeline_config"""
    return get_training_pipeline_config()


class DataIngestionConfig:
    def __init__(self):
        from src.constants import (
            DATA_INGESTION_DIR_NAME, DATA_INGESTION_FEATURE_STORE_DIR,
            DATA_INGESTION_INGESTED_DIR, FILE_NAME, TRAIN_FILE_NAME, TEST_FILE_NAME,
            DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO, DATA_INGESTION_COLLECTION_NAME
        )
        tpc = get_training_pipeline_config()
        self.data_ingestion_dir = os.path.join(tpc.artifact_dir, DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
        self.training_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
        self.testing_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
        self.train_test_split_ratio = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name = DATA_INGESTION_COLLECTION_NAME
