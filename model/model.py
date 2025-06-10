### Imports
# Local
from encoder import Encoder
from decoder import Decoder

# External
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


### GRAPH-ETM MODEL ARCHITECTURE
## Main GraphETM model.
# Description: MODEL assembles the GraphETM architecture using the ENCODER and DECODER blocks.

# ------------------------------------------------------------------
# @title GraphETM Model