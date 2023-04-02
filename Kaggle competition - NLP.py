# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.metrics import f1_score
import os
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


# load the data
train_df = pd.read_csv('C:\\Users\\tony3\\Desktop\\nlp-getting-started\\train.csv')
test_df  = pd.read_csv('C:\\Users\\tony3\\Desktop\\nlp-getting-started\\test.csv')

#Data Preprocessing
#=============================================================================================================
# data preprocessing
train_df.drop(['id', 'keyword', 'location'], axis=1, inplace=True)
test_df.drop([ 'keyword', 'location'], axis=1, inplace=True)

# remove any URLs or links from the text
train_df['text'] = train_df['text'].apply(lambda x: ' '.join([word for word in x.split() if 'http' not in word]))
test_df['text'] = test_df['text'].apply(lambda x: ' '.join([word for word in x.split() if 'http' not in word]))

# remove any punctuation
train_df['text'] = train_df['text'].str.replace('[^\w\s]','', regex=True)
test_df['text'] = test_df['text'].str.replace('[^\w\s]','', regex=True)

# remove any non-alphabetic characters
train_df['text'] = train_df['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
test_df['text'] = test_df['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

# convert all text to lowercase
train_df['text'] = train_df['text'].apply(lambda x: x.lower())
test_df['text'] = test_df['text'].apply(lambda x: x.lower())

# tokenize the text into individual words
train_df['text'] = train_df['text'].apply(lambda x: x.split())
test_df['text'] = test_df['text'].apply(lambda x: x.split())

# remove any stop words
stop_words = set(stopwords.words('english'))
train_df['text'] = train_df['text'].apply(lambda x: [word for word in x if word not in stop_words])
test_df['text'] = test_df['text'].apply(lambda x: [word for word in x if word not in stop_words])

#=============================================================================================================

# # use the Word2Vec model to remove any words that are not in the model's vocabulary
# model = Word2Vec(sentences=train_df['text'], vector_size=100, window=4, workers=4, min_count=1, epochs=10)
#
# # build the vocabulary from the training data
# model.build_vocab(train_df['text'])
#
# # train the Word2Vec model on the training data
# model.train(train_df['text'], total_examples=model.corpus_count, epochs=10)


glove_input_file = 'C:\\Users\\tony3\\Desktop\\nlp-getting-started\\glove.twitter.27B.100d.txt'
word2vec_output_file = 'glove.twitter.27B.100d.word2vec.txt'


def load_glove_vectors(glove_file):
    embeddings_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
    return embeddings_dict

embeddings_dict = load_glove_vectors(glove_input_file)


def get_vectors(corpus, embeddings_dict, vector_size):
    vectors = []

    for doc in corpus:
        doc_vector = np.zeros(vector_size)
        num_words = 0
        for word in doc:
            if word in embeddings_dict:
                doc_vector += embeddings_dict[word]
                num_words += 1
        if num_words:
            doc_vector /= num_words
        vectors.append(doc_vector)

    return vectors


# create feature vectors for each tweet
train_vectors = get_vectors(train_df.text, embeddings_dict, 100)
test_vectors = get_vectors(test_df.text, embeddings_dict, 100)

# convert the feature vectors into a dataframe
train_target = train_df['target']
train_df = pd.DataFrame(train_vectors)
test_df = pd.DataFrame(test_vectors)



# # rejoin the words back into a single string
# train_df['text'] = train_df['text'].apply(lambda x: ' '.join(x))
# test_df['text'] = test_df['text'].apply(lambda x: ' '.join(x))

# split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(train_vectors, train_target, test_size=0.2, random_state=42)

# # use GridSearchCV to find the best parameters for the model
# param_grid = {'vect__max_df': [0.5, 0.75, 1.0], 'vect__max_features': [1000, 5000, 10000], 'tfidf__use_idf': [True, False], 'clf__C' : [0.1, 1, 10], 'clf__penalty': ['l1', 'l2']}

# create a pipeline to vectorize the text, transform it into a tf-idf matrix, and then train a logistic regression model
#pipeline = Pipeline([('vect', CountVectorizer(max_df=0.5, max_features=5000,  ngram_range=(1, 2))),('tfidf', TfidfTransformer(use_idf=True)),('clf', LogisticRegression(C=1, penalty='l2'))])
pipeline = Pipeline([('clf', LogisticRegression(C=1, penalty='l2'))])
# # create a GridSearch object with the pipeline and hyperparameters
# grid_search = GridSearchCV(pipeline, param_grid = param_grid, cv=5, n_jobs=-1, verbose=1, scoring='f1')
#
# # fit the model
# grid_search.fit(X_train, y_train)
#
# # print the best parameters
# print('Best hyperparameters: ', grid_search.best_params_)
# print('Best score: ', grid_search.best_score_)

# output
# Best hyperparameters:  {'clf__C': 1, 'clf__penalty': 'l2', 'tfidf__use_idf': True, 'vect__max_df': 0.5, 'vect__max_features': 5000}
# Best score:  0.7473164719710865

# fit the model
pipeline.fit(X_train, y_train)

# make predictions on the test set
y_pred = pipeline.predict(X_test)

# print the accuracy score
print(' Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print(' Classification Report: {}'.format(classification_report(y_test, y_pred)))
print(' Confusion Matrix: {}'.format(confusion_matrix(y_test, y_pred)))
print(' f1-score: {}'.format(f1_score(y_test, y_pred)))

# # make predictions on the test set
# test_pred = pipeline.predict(test_df['text'])
#
# # create a submission file
# submission = pd.DataFrame({'id': test_df['id'], 'target' : test_pred})
# submission.to_csv('submission.csv', index=False)
#
# # print the first 5 rows of the submission file
# print(submission.head())
#
# # print the directory where the submission file is saved
# print(os.getcwd())
#
#
