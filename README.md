# Product Recomendation: Building a Robust Recommender

Abstract: Product recommenders are a key structure to many industries. People are searching for valuable recomendations they can use, and companies are looking to optimize profits. Using Amazon consumer reviews collected from Julian McAuley at UCSD, I focused on building a robust recommender to help consumers discover products they would enjoy. Focusing on Health and Personal Care, I build and test collaberative and hybrid recommenders.

Results: None to report yet.

See this work as a presentation in [Google Slides](https://docs.google.com/presentation/d/1HuLg7flwSoy_YKFmS6S6ypa1kuKpDmKfMoaDjzYe5xc/edit?usp=sharing).

[See the video](https://youtu.be/6SmLwANBp_4) of this talk.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=6SmLwANBp_4" target="_blank"><img src="http://img.youtube.com/vi/6SmLwANBp_4/0.jpg" 
alt="Theodore Speaks - How to Persuade and Inspire Like TED" width="240" height="180" border="10" /></a>

# Background & Motivation
Recommenders are truely collaberative in nature. People are looking for products they will enjoy, venders are looking to better connect with consumers and sell their products. Outside of product sales, recommenders are very flexible and can be used in a variety of environments. It is important to create robust models which are capable of capturing the complexity of the user and the recommended product.

We live in a world with more choices than ever at our finger tips, often leading to the paradox of choice: too many choices create an inability to make a choice. Personalized recommendations can fix this.

## The Problem
How to deliver the right product to the right person?

We live in a world with more choices than ever at our finger tips, often leading to the paradox of choice: too many choices create an inability to make a choice. 

## The Solution

Personalized recommendations an end user can understand. Connect product to person.

---

# Analysis methods

Libraries and modules include ```Python3```, ```NumPy```, ```Pandas```, ```Surprise```, ```LightFM```, ```Seaborn```, ```Keras```.

The tech stack consists of Python 3, Numpy, Pandas, Beautiful Soup, Linguistic Inquiry and Word Count (LIWC), Natural Language Toolkit (NLTK), Scikit-Learn, Matplotlib, HTML, CSS, Tableau, Flask, and Heroku.

From the ```src``` directory of the repo, 

Two ```csv``` files, the results of the webscraping, are stored in the ```data``` directory. 

```ted-main.csv``` has the metadata for 2638 TED Talks- all talks featured on TED.com from 2006 through 2017.
```transcripts.csv``` contains the transcripts for 2542 talks - the transcripts are not available for every talk.

Four text transcript files are also stored in the data directory. These transcripts cannot be stored in a CSV because they are larger than the 32,767 character limit for a cell.


To prepare the dataset for analyses:

From the ```src``` directory of the repo, run the following code:

```python assemble.py```

```python annotate.py```

```python process-text.py```

These scripts: 
- join large transcripts to dataframe for analysis
- drop rows with missing transcripts
- remove talks centered around music performances
- remove talks with more than 1 speaker
- create features like 'applause', 'laughter' from transcript
- normalize ratings counts to account for number of times the talk has been viewed
- divide transcripts into halves and quarters
- add results of LIWC analysis and create emotion word change features 

Edits to transcripts were done by script and by hand to remove question and answer sections and conversations with multiple speakers.

If structural changes to the cleaning and feature engineering are required, rerun the results of ```annotate.py```, the dataset in ```all_after_annotate.xls```, through LIWC module to produce per document word category ratios. A license with LIWC is required and is available at [liwc.net](http://www.liwc.net)</a>.

After running the 3 scripts above, you have a final dataset ```all_with_liwc_segmented.xls``` with features ready for statistical models (93.5 MB).




For all the following analyses, the response variable is set in the ```settings.py``` file, on line 3, under the variable name "TARGET".

For response variables, you might choose from 'norm_persuasive', 'norm_inspiring', 'views', 'comments', or 'applause'.

To fit a decision tree, and see the top feature importances, run:

```python predict-decision-tree.py```

To fit a random forest regressor and see the top feature importances, run:

```python predict-random-forest.py```

To build a linear regression model with most important features from the previous steps as predictors, run:

```python predict-linear.py```



To explore the 10 primary components in TED Talks using non-negative matrix factorization to perform clustering, run:

```python clustering.py``` 

To train a classifier model to predict 'persuasive' and 'non-persuasive' texts, run:

```python classification.py```

You can also access this classifier tool by visiting [theodorespeaks.com](http://www.theodorespeaks.com), scrolling down, and inputting your own text into the text box and hitting "Submit".
The page will reload with a "Persuasive" or "Non-Persuasive" prediction with a probability beside the text box. 

To find a similar TED speaker based on Euclidean distance and linguistic feature similarity to a speaker you specify,
 change the SPEAKER_FULL_NAME variable in line 11 and run: 

```python distance.py```

You can also access this "Find a Similar Speaker" tool by visiting [theodorespeaks.com](http://www.theodorespeaks.com), scrolling down, and inputting a speaker's full name into the text box and hitting "Submit".
