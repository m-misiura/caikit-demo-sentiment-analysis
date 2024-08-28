import os
from . import data_model, runtime_model
import caikit


CONFIG_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), "config.yml"))
caikit.configure(CONFIG_PATH)
