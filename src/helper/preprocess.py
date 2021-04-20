import string
import re
import gzip
import pandas as pd
import itertools
import numpy as np
from lightfm import LightFM



def parse(path):
    """
    Opens gzipped json for getDF to read
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
    helper function, maps vert_high, high, medium, low to count of apps reviewed
    """
    if row >= 133:
        return 'very_high'
    elif row < 133 and row >= 41:
        return 'high'
    elif row < 41 and row >= 18:
        return 'medium'
    else:
        return 'low'


    
class preprocess_pipeline:
    
    """
    Preprocessing Pipeline Class to merge and clean datasets for production
    
    Arguments
    ---------
    user_data: csv format, must include overall score for product
    
    meta_data: csv format, meta data for each product in category
    
    """
    
    def __init__ (self, user_data, meta_data):
        self.user_data = user_data
        self.meta_data = meta_data
        
        
    def merge_user_meta(self):
        
        """
        Merge user data and meta data into combined dataframe
        """
        
        self.merged = self.user_data.merge(self.meta_data, how='left', on='asin')
        
        
        
    def unpack(self, row):
        
        '''
        helper function for unpack_sales_rank, unpacks the dictionary inside salesRank
        
        
        Arguments
        ---------
        row: spesific row of merged dataframe
        
        Returns
        ---------
        returns value inside salesRank dictionary unless there is no dictionary present,
        0 will be returned.
        
        '''
        
        if isinstance(row, dict):
            for k, v in row.items():
                return v
        else:
            0
    
    def unpack_sales_rank(self, null=False):
        
        """
        unpacks sales rank dictionary with helper function unpack. If no avalible ranking information,
        will drop column
        
        Arguments
        ---------
        null: indicator if row is completely empty
        
        Returns
        ---------
        Value inside dictionary if null is false, else will drop the salesRank column
        
        
        """
        if null == False:
            self.merged.salesRank = self.merged.salesRank.apply(lambda x: self.unpack(x))
            self.merged.salesRank.fillna(0, inplace=True)
        else:
            self.merged.drop('salesRank', axis=1, inplace=True)

    
    def set_quantile(self):
        
        """
        Counts product review quantiles and creates a column categorizing products accordingly
        
        
        Returns
        -------
        Quantile pandas series, mapping according to quantile list
        
        """
        
        # select items and group them by their review couunt
        review_count = self.merged.groupby('asin').count().sort_values(by='reviewerID', ascending=False)['reviewerID']

        # list of qantiles desired
        quartile_list = review_count.quantile([0.25, 0.75, 0.9, 0.99]).tolist()
    
        # populates a dictionary according to the mapped quartile list with review_map helper function
        prod_review_count_dict = dict(review_count.apply(lambda x: review_map(x)))
        
        # create new column with products mapped by quantile
        self.merged['prod_review_counts'] = self.merged['asin'].apply(lambda x: prod_review_count_dict[x])
    
    def clean_categories(self):
        
        """
        preprocesses category column to extract best information
        """
        
        # flatten nested lists
        self.merged.categories = self.merged.categories.apply(lambda x: list(itertools.chain(*x)))
        
        # remove redundant information
        self.merged.categories = self.merged.categories.apply(lambda x: " ".join(x[1:]))
        
        # remove puncuation
        self.merged.categories = self.merged.categories.apply(
            lambda x: x.translate(str.maketrans('', '', string.punctuation))
        )
        
        # remove redundant words
        self.merged.categories = self.merged.categories.apply(lambda x: "-".join(list(set(x.lower().split()))))

        
    def prep(self, null=False):
        
        self.merge_user_meta()
        self.unpack_sales_rank(null)
        self.clean_categories()
        self.set_quantile()
        
        return self.merged
        
    
        
        
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