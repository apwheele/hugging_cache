'''
Functions to calculate similarity
between two strings based on NLP
model in simpletransformers
'''

from functools import lru_cache
import numpy as np
import os
import pandas as pd
from simpletransformers.language_representation import RepresentationModel

cache_loc = "./cache"
if os.path.exists(cache_loc):
    print('Cache exists, loading pre-downloaded model')
else:
    print('Cache does not exist, downloading model')

model = RepresentationModel("bert","ProsusAI/finbert",cache_dir=cache_loc)

# memoizing this part, much more expensize than norm
@lru_cache
def string_embed(x,mod=model):
    return model.encode_sentences([x],combine_strategy="mean")

# Intended to be used in pandas dataframe, looks
# at differences in two strings
def sim_strings(x):
    x0 = string_embed(x[0])
    x1 = string_embed(x[1])
    return np.linalg.norm(x0 - x1,ord=2)