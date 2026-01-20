import os
from datetime import datetime
from dataclasses import dataclass
from src.constants import (
    MODEL_FILE_NAME,
    PREPROCSSING_OBJECT_FILE_NAME,
    MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE,
    MODEL_BUCKET_NAME,
)

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


class DataValidationConfig:
    def __init__(self):
        from src.constants import (
            DATA_VALIDATION_DIR_NAME, DATA_VALIDATION_REPORT_FILE_NAME
        )
        tpc = get_training_pipeline_config()
        self.data_validation_dir = os.path.join(tpc.artifact_dir, DATA_VALIDATION_DIR_NAME)
        self.validation_report_file_path = os.path.join(self.data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)


class DataTransformationConfig:
    def __init__(self):
        from src.constants import (
            DATA_TRANSFORMATION_DIR_NAME, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, TRAIN_FILE_NAME, TEST_FILE_NAME
        )
        tpc = get_training_pipeline_config()
        self.data_transformation_dir = os.path.join(tpc.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                        TRAIN_FILE_NAME.replace("csv", "npy"))
        self.transformed_test_file_path = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                       TEST_FILE_NAME.replace("csv", "npy"))
        self.transformed_object_file_path = os.path.join(self.data_transformation_dir,
                                                         DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                         PREPROCSSING_OBJECT_FILE_NAME)


class ModelTrainerConfig:
    def __init__(self):
        from src.constants import (
            MODEL_TRAINER_DIR_NAME, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME,
            MODEL_TRAINER_EXPECTED_SCORE, MODEL_TRAINER_MODEL_CONFIG_FILE_PATH,
            MODEL_TRAINER_N_ESTIMATORS, MODEL_TRAINER_MIN_SAMPLES_SPLIT, MODEL_TRAINER_MIN_SAMPLES_LEAF,
            MODEL_TRAINER_MAX_DEPTH, MODEL_TRAINER_CRITERION, MODEL_TRAINER_RANDOM_STATE
        )
        tpc = get_training_pipeline_config()
        self.model_trainer_dir = os.path.join(tpc.artifact_dir, MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path = os.path.join(self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME)
        self.expected_accuracy = MODEL_TRAINER_EXPECTED_SCORE
        self.model_config_file_path = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
        self._n_estimators = MODEL_TRAINER_N_ESTIMATORS
        self._min_samples_split = MODEL_TRAINER_MIN_SAMPLES_SPLIT
        self._min_samples_leaf = MODEL_TRAINER_MIN_SAMPLES_LEAF
        self._max_depth = MODEL_TRAINER_MAX_DEPTH
        self._criterion = MODEL_TRAINER_CRITERION
        self._random_state = MODEL_TRAINER_RANDOM_STATE


class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME
    
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME