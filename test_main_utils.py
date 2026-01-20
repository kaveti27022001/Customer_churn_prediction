import sys
import traceback

print("Step 1: Import numpy and dill")
try:
    import numpy as np
    import dill
    print("✓ numpy and dill imported")
except Exception as e:
    print("✗ Failed")
    traceback.print_exc()

print("\nStep 2: Import exception")
try:
    from src.exception import MyException
    print("✓ MyException imported")
except Exception as e:
    print("✗ Failed")
    traceback.print_exc()

print("\nStep 3: Import logger")
try:
    from src.logger import logging
    print("✓ logging imported")
except Exception as e:
    print("✗ Failed")
    traceback.print_exc()

print("\nStep 4: Now try to import main_utils")
try:
    from src.utils import main_utils
    print(f"✓ main_utils imported")
    print(f"  Functions: {[x for x in dir(main_utils) if not x.startswith('_')]}")
except Exception as e:
    print("✗ Failed to import main_utils")
    traceback.print_exc()

print("\nStep 5: Execute the file directly")
try:
    exec(open('src/utils/main_utils.py').read())
    print("✓ File executed successfully")
except Exception as e:
    print("✗ Failed to execute")
    traceback.print_exc()
