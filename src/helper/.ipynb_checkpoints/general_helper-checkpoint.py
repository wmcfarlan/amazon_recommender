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
    if row >= 133:
        return 'very_high'
    elif row < 133 and row >= 41:
        return 'high'
    elif row < 41 and row >= 18:
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
    ll = ['prod_review_counts:', 'salesRank:', 'price:']
    aa = my_list
    for x, y in zip(ll, aa):
        res = str(x) +""+ str(y)
        result.append(res)
    return result




def unpack(row):
    if isinstance(row, dict):
        for k, v in row.items():
            return v
    else:
        0
        

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