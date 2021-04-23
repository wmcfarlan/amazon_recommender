import pandas as pd
import numpy as np
from collections import defaultdict
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class custom_predictor:

    """
    This class runs a selected ML regressor in order to predict user
    ratings based on selected NLP features
    """
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
#     def get_target_col(self):
#         self.target_column = self.dataframe.groupby('asin').count().sort_values(
#             by='reviewerID', ascending=False)['reviewerID'].index[0]

#     def fill_asin_title_dict(self):
        
#         asin_title = self.dataframe[['asin', 'title']]
#         asin_title.drop_duplicates()['title']
#         asin_title.index = asin_title.asin
        
#         self.asin_title_dict = dict(asin_title['title'])
#         self.title_asin_dict = {v:k for k, v in self.asin_title_dict.items()}
    
    def set_index(self):
        
        """
        sets index as user ID and drops ID column from dataframe
        """
        
        self.dataframe.index = self.dataframe.reviewerID
        self.dataframe.drop('reviewerID', axis=1, inplace=True)
    
    
    def populate_dictionary(self, text_columns):
        
        """
        user selected NLP columns are passed into a created dictionary
        
        Arguments
        ---------
        text_columns: list, a list of all text based columns that want to be passed into the model
        
        """
        
        # set empty default dictionary
        self.nlp_dict = defaultdict(list)
        
        # get unique IDs
        self.unique_ids = self.dataframe.index.unique()
        
        # populate dictionary with unique IDs
        for id in self.unique_ids:
            self.nlp_dict[id]
            
        # populate dictionary with text features
        for text in text_columns:
            for idx, text in zip(self.dataframe.index, self.dataframe[text]):
                self.nlp_dict[idx].append(text)
    
    
    def get_targets(self):
        
        """
        Creates pivot matrix with index as user id, columns as product ids, and values as user ratings
        """
        
        self.pivot_matrix = self.dataframe.pivot_table(index='reviewerID', columns='asin', values='overall')
        self.target_list = self.pivot_matrix.columns
    
    def nlp_prep(self, sentence):
        
        """
        Helper function for text cleaning in create_nlp_df function
        
        Arguments
        ---------
        sentence: list, a list filled with all text data
        
        """
        
        sentence = " ".join(sentence)
        sentence = sentence.lower()
        cleaner = re.compile('<.*?>')
        clean_text = re.sub(cleaner, '', sentence)
        rem_num = re.sub('[0-9]+', '', clean_text)
        return rem_num
        
    
    def create_nlp_df(self):
        
        """
        creates a tfidf dataframe with preprocessed text features
        """
        
        self.nlp_df = pd.DataFrame(list(self.nlp_dict.items()), columns=['user', 'text_data'])
        
        self.nlp_df.index = self.nlp_df.user
        self.nlp_df.drop('user', axis=1, inplace=True)
        
        self.nlp_df.text_data = self.nlp_df.text_data.apply(lambda x: self.nlp_prep(x))
        
        vectorizer = TfidfVectorizer(stop_words='english')
        vect = vectorizer.fit_transform(self.nlp_df.text_data)
        
        #testing np
        self.vect_df = pd.DataFrame(data=vect.todense(), index=self.nlp_df.index, columns=vectorizer.get_feature_names())

            
    
    def fit(self, text_columns):
        
        """
        fit the model to generate proper data for prediction
        """
        
#         self.fill_asin_title_dict()
        self.set_index()
        self.populate_dictionary(text_columns)
        self.get_targets()
        self.create_nlp_df()
    
    
    
    def predict_one_eff(self, model, product):
        
        """
        Creates predictions for each item in pivot_df space to fill out missing data
        
        
        Arguments
        ---------
        model: sklearn regressor, the parameters can be user selected
        
        
        Returns
        ---------
        Returns a dictionary with the name of each product and predictions for each user
        
        """
        
        
        model = model
        pred_dict = defaultdict(int)
        
        full_set = self.vect_df.merge(self.pivot_matrix.loc[:, product], left_index=True, right_index=True)
        full_set.fillna(0, inplace=True)
        full_set = full_set.to_numpy()
        
        filled_data = full_set[full_set[:, -1] != 0]
        
        X = filled_data[:, :-1]
        y = filled_data[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        model.fit(X_train, y_train)
        preds_test = model.predict(X_test)
        
        error = mean_squared_error(y_test, preds_test)
        
        print(error)
        
        true_preds = model.predict(full_set[:, :-1])
        pred_dict[product] = true_preds
        
        return pred_dict
    
    
    
    def predict_one(self, model, product):
        
        """
        Creates predictions for each item in pivot_df space to fill out missing data
        
        
        Arguments
        ---------
        model: sklearn regressor, the parameters can be user selected
        
        
        Returns
        ---------
        Returns a dictionary with the name of each product and predictions for each user
        
        """
        
        
        model = model
        pred_dict = defaultdict(int)
        
        full_set = self.vect_df.merge(self.pivot_matrix.loc[:, product], left_index=True, right_index=True)
        full_set.fillna(0, inplace=True)
        filled_data = full_set[full_set[product] != 0]
        
        X = filled_data.iloc[:, :-1]
        y = filled_data.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        model.fit(X_train, y_train)
        preds_test = model.predict(X_test)
        
        error = mean_squared_error(y_test, preds_test)
        
        print(error)
        
        true_preds = model.predict(full_set.iloc[:, :-1])
        pred_dict[product] = true_preds
        
        return pred_dict
    
    
    def predict_all(self, model):
        
        model = model
        error_dict = defaultdict(int)
        pred_dict = defaultdict(int)
        for col in self.target_list:
            full_set = self.vect_df.merge(self.pivot_matrix.loc[:, col], left_index=True, right_index=True)
            full_set.fillna(0, inplace=True)
            filled_data = full_set[full_set[col] != 0]
            
            X = filled_data.iloc[:, :-1]
            y = filled_data.iloc[:, -1]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            
            
            model.fit(X_train, y_train)
            preds_test = model.predict(X_test)
            
            error = mean_squared_error(y_test, preds_test)
            error_dict[col] = error
            
            preds = model.predict(full_set.iloc[:, :-1])
            pred_dict[col] = preds
        return pred_dict, error_dict
            

        
        
        
class cluster_rec:
    
    def __init__ (self):
        pass
        
        
        
if __name__ == "__main__":
    print('main')
        
    