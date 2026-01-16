import os
from datetime import date

# Project Information
PROJECT_NAME = "Customer-Churn-Prediction"
PROJECT_DESCRIPTION = "Telco Customer Churn Prediction using Machine Learning"

# For MongoDB connection
DATABASE_NAME = "Customer-Churn"
COLLECTION_NAME = "Proj1-Data"
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = "customer_churn_pipeline"
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "customer_churn_model.pkl"
SCALER_FILE_NAME = "scaler.pkl"
ENCODERS_FILE_NAME = "encoders.pkl"

TARGET_COLUMN = "Churn"
CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "Proj1-Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME: str = "report.html"

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_PREPROCESSED_OBJECT_FILE_NAME: str = "preprocessor.pkl"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.7
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float = 0.05
MODEL_TRAINER_N_ESTIMATORS: int = 200
MODEL_TRAINER_MIN_SAMPLES_SPLIT: int = 7
MODEL_TRAINER_MIN_SAMPLES_LEAF: int = 6
MODEL_TRAINER_MAX_DEPTH: int = 10
MODEL_TRAINER_CRITERION: str = "gini"
MODEL_TRAINER_RANDOM_STATE: int = 101

"""
MODEL EVALUATION related constants
"""
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_FILE_NAME: str = "evaluation_report.yaml"

"""
MODEL PUSHER related constants
"""
MODEL_PUSHER_DIR_NAME: str = "model_pusher"
MODEL_BUCKET_NAME = "customer-churn-model-registry"
MODEL_PUSHER_S3_KEY = "model-registry"

"""
Application Configuration
"""
APP_HOST = "0.0.0.0"
APP_PORT = 8000
LOG_DIR = "logs"