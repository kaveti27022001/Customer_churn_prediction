# below code is to check the logging config
from src.logger import logging

logging.debug("This is a debug message.")
logging.info("This is an info message.")
logging.warning("This is a warning message.")
logging.error("This is an error message.")
logging.critical("This is a critical message.")

#----------------------------------------------------------------

from src.pipline.training_pipeline import TrainPipeline

try:
    print("\n" + "="*80)
    print("STARTING ML PIPELINE")
    print("="*80 + "\n")
    
    pipline = TrainPipeline()
    pipline.run_pipeline()
    
    print("\n" + "="*80)
    print("ML PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
except Exception as e:
    print(f"\n{'='*80}")
    print(f"PIPELINE FAILED WITH ERROR:")
    print(f"{'='*80}")
    print(f"Error: {str(e)}")
    print(f"{'='*80}\n")
    raise