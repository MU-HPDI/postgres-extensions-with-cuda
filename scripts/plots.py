import pandas as pd
import sys
import os

# Get the absolute path of the parent directory of this script
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'scripts'))

import db_funcs as db
