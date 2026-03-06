# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

from mmseg.registry import MODELS as SEG_MODELS

from triplane_encoder import *


def build(model_config):
    model = SEG_MODELS.build(model_config)
    model.init_weights()
    return model
