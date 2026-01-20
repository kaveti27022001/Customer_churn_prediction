import sys
import traceback

# Test each import separately
imports_to_test = [
    ("src.exception", "MyException"),
    ("src.logger", "logging"),
    ("src.utils.main_utils", "load_numpy_array_data, load_object, save_object"),
    ("src.entity.config_entity", "ModelTrainerConfig"),
    ("src.entity.artifact_entity", "DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact"),
    ("src.entity.estimator", "MyModel"),
]

for module, items in imports_to_test:
    try:
        exec(f"from {module} import {items}")
        print(f"✓ Successfully imported {items} from {module}")
    except Exception as e:
        print(f"✗ Failed to import {items} from {module}")
        traceback.print_exc()
        print()
