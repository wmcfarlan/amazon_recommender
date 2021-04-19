import gzip
import pandas as pd
import itertools
import numpy as np
from lightfm import LightFM

def parse(path):
    """
    Opens gzipped file for getDF to read
    """
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    """
    getDF will use parse to fill a dictionary with key value pairs in parsed json file
    
    Returns
    -------
    Pandas Dataframe object
    """
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def review_map(row):
    """
    helper function, maps high, medium, low to count of apps reviewed
    """
    
    if row >= 41:
        return 'high'
    if row >= 14 and row < 41:
        return 'medium'
    else:
        return 'low'
    
    
def feature_colon_value(my_list):
    """
    Takes as input a list and prepends the columns names to respective values in the list.
    For example: if my_list = [1,1,0,'del'],
    resultant output = ['f1:1', 'f2:1', 'f3:0', 'loc:del']
   
    """
    result = []
    ll = ['free:','prod_review_counts:', 'categories:']
    aa = my_list
    for x, y in zip(ll, aa):
        res = str(x) +""+ str(y)
        result.append(res)
    return result




def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    while True:
        yield {
            "no_components": np.random.randint(16, 100),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp", "warp-kos"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-8),
            "max_sampled": np.random.randint(5, 15),
            "num_epochs": np.random.randint(5, 100),
        }
        


if __name__ == '__main__':
    print('main')
    
    
#             """
#         Info HERE
        
#         Arguments
#         ---------

#         arg_1: just info
#                 about args
#         arg_2: more info

#         Returns
#         -------

#         here is what is
#             retruned
#         """