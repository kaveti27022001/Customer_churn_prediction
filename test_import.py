import sys
import traceback

try:
    from src.entity.artifact_entity import ClassificationMetricArtifact
    print("Success!")
except Exception as e:
    print("Error importing ClassificationMetricArtifact:")
    traceback.print_exc()
    
print("\nModule contents:")
import src.entity.artifact_entity as ae
print(dir(ae))
