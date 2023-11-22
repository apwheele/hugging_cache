'''
Example caching huggingface
model in github actions pipeline
'''

import pandas as pd
from src.sim_funcs import cache_loc, sim_strings

a = ['Statement about money']*3
b = ['Statement about $$$','something different','this is a invoice for a $1,000 dollars']

df = pd.DataFrame(zip(a,b),columns=['a','b'])
df['SimScore'] = df.apply(sim_strings,axis=1)
print(df)