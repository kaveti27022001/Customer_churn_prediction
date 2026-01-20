import sys
import traceback

try:
    from src.components.model_trainer import ModelTrainer
    print("Success!")
except Exception as e:
    print("Error importing ModelTrainer:")
    traceback.print_exc()
    
print("\nModule contents:")
import src.components.model_trainer as mt
print(dir(mt))
