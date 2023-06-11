# from .models import *
# from .papyrus import *
# from .run_scripts import *

# from .utils import *
# from .chemutils import *
# from .papyrus import *
# from .baselines import *
# from .models import *
# from .run_scripts import *

import os

# Get the absolute path to the project's root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the paths to different directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'run_scripts')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Define the paths to different files
# CONFIG_FILE = os.path.join(BASE_DIR, 'config.yaml')
# REQUIREMENTS_FILE = os.path.join(BASE_DIR, 'requirements.txt')

# Export the paths as variables for easy access in other modules
os.environ['BASE_DIR'] = BASE_DIR
os.environ['DATA_DIR'] = DATA_DIR
os.environ['LOGS_DIR'] = LOGS_DIR
os.environ['CONFIG_DIR'] = CONFIG_DIR
os.environ['SCRIPTS_DIR'] = SCRIPTS_DIR
os.environ['MODELS_DIR'] = MODELS_DIR
# os.environ['CONFIG_FILE'] = CONFIG_FILE
# os.environ['REQUIREMENTS_FILE'] = REQUIREMENTS_FILE

__all__ = ['BASE_DIR', 'DATA_DIR', 'LOGS_DIR', 'CONFIG_DIR', 'SCRIPTS_DIR', 'MODELS_DIR',
           # 'models', 'papyrus', 'run_scripts'
           ]

