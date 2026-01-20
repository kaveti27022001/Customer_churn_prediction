import sys
sys.path.insert(0, '.')

from src.components.data_validation import DataValidation
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataValidationConfig

# Use existing data artifact
artifact = DataIngestionArtifact(
    trained_file_path='artifact/01_17_2026_16_10_21/data_ingestion/ingested/train.csv',
    test_file_path='artifact/01_17_2026_16_10_21/data_ingestion/ingested/test.csv'
)

# Get validation config
config = DataValidationConfig()

# Run validation
validation = DataValidation(artifact, config)
result = validation.initiate_data_validation()

print("\n" + "="*80)
print("DATA VALIDATION RESULTS")
print("="*80)
print(f"Status: {result.validation_status}")
print(f"Message: {result.message}")
print(f"Report: {result.validation_report_file_path}")
print("="*80)
