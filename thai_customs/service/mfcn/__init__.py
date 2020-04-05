
"""Top-level package for Text Mountain."""

import os
import sys

from . import src, agents, datasets, graphs, utils

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
