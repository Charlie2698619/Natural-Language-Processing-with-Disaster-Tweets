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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns



# load the data
train_df = pd.read_csv('C:\\Users\\tony3\\Desktop\\nlp-getting-started\\train.csv')
test_df  = pd.read_csv('C:\\Users\\tony3\\Desktop\\nlp-getting-started\\test.csv')
test_ids = test_df['id']


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

# create a Word2Vec model
glove_input_file = 'C:\\Users\\tony3\\Desktop\\nlp-getting-started\\glove.twitter.27B.100d.txt'
word2vec_output_file = 'glove.twitter.27B.100d.word2vec.txt'

# convert the glove file into a word2vec file
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

# create a function to create feature vectors for each tweet
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


# split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(train_vectors, train_target, test_size=0.2, random_state=42)

#=============================================================================================================
# # try multiple models
# models = [ ('Logistic Regression', LogisticRegression(C=10, penalty='l2')), ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)), ('SVM', SVC(kernel= 'linear', C=1)), ('KNN', KNeighborsClassifier(n_neighbors=5)),]
#
# # def function to run the models
# def run_models(models, X_train, y_train, X_test, y_test):
#     for name, model in models:
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         print(name, 'F1 score: ', f1_score(y_test, y_pred))
#
# # run the models
# run_models(models, X_train, y_train, X_test, y_test)
#
# # # output
# # Logistic Regression F1 score:  0.7465362673186634
# # Random Forest F1 score:  0.7303465765004227
# # SVM F1 score:  0.7387387387387386
# # KNN F1 score:  0.7507739938080494
#=============================================================================================================
# find the best classifier by using ensemble voting
# classifiers = [
#     ('KNN', KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='auto', leaf_size=90, p=2, metric='manhattan')),
#     ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
#     ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=42)),
#     ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
# ]
#
# # fit and score the classifiers
# def fit_and_score(classifiers, X_train, y_train, X_test, y_test):
#     np.random.seed(42)
#     model_scores = {}
#     for name, model in classifiers:
#         model.fit(X_train, y_train)
#         model_scores[name] = model.score(X_test, y_test)
#     return model_scores
#
# # print the scores
# model_scores = fit_and_score(classifiers, X_train, y_train, X_test, y_test)
# print(model_scores)
# # # output
# #{'KNN': 0.7971109652002626, 'Random Forest': 0.8010505581089954, 'AdaBoost': 0.7957977675640184, 'Gradient Boosting': 0.8023637557452397}
#
# # create a voting classifier
# voting_clf = VotingClassifier(estimators=classifiers, voting='hard', n_jobs=-1)
# voting_clf.fit(X_train, y_train)
# y_pred = voting_clf.predict(X_test)
# print('Voting Classifier F1 score: ', f1_score(y_test, y_pred))
# # # output
# # f1-score: 0.7572663000785547


#=============================================================================================================
# # use RandomSearchCV to find the best parameters for the model
# param_grid = {
#     'GradientBoosting__n_estimators': [10, 50, 100, 150, 200],
#     'GradientBoosting__learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'GradientBoosting__max_depth': [3, 4, 5, 6, 7],
#     'GradientBoosting__min_samples_split': [2, 3, 4],
#     'GradientBoosting__min_samples_leaf': [1, 2, 3],
#     'GradientBoosting__subsample': [0.8, 0.9, 1.0],
#     'GradientBoosting__max_features': [None, 'sqrt', 'log2']
# }
#
#
# # create a pipeline to vectorize the text, transform it into a tf-idf matrix, and then train a logistic regression model
# pipeline = Pipeline([('GradientBoosting', GradientBoostingClassifier(random_state=42))])
# # create a GridSearch object with the pipeline and hyperparameters
# random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv=5, n_iter=20, verbose=1, n_jobs=-1, scoring='f1', random_state=42)
#
# # fit the model
# random_search.fit(X_train, y_train)
#
# # print the best parameters
# print('Best hyperparameters: ', random_search.best_params_)
# print('Best score: ', random_search.best_score_)
# #
# # # output
# Best hyperparameters:  {'GradientBoosting__subsample': 0.8, 'GradientBoosting__n_estimators': 150, 'GradientBoosting__min_samples_split': 3, 'GradientBoosting__min_samples_leaf': 2, 'GradientBoosting__max_features': 'sqrt', 'GradientBoosting__max_depth': 6, 'GradientBoosting__learning_rate': 0.05}
# Best score:  0.7620484697332813

# #=============================================================================================================
# pipeline with best parameters
pipeline = Pipeline([( 'GradientBoosting', GradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_features='sqrt', min_samples_leaf=2, min_samples_split=3, n_estimators=150, random_state=42, subsample=0.8))])

# fit the model
pipeline.fit(X_train, y_train)

# make predictions on the test set
y_pred = pipeline.predict(X_test)

# print the accuracy score
print(' Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print(' Classification Report: {}'.format(classification_report(y_test, y_pred)))
print(' Confusion Matrix: {}'.format(confusion_matrix(y_test, y_pred)))
print(' f1-score: {}'.format(f1_score(y_test, y_pred)))

#=============================================================================================================
# model evaluation
def plot_confusion_matrix(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cbar=False)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_test, y_pred)

#=============================================================================================================
# make predictions on the test set
test_pred = pipeline.predict(test_vectors)

# create a submission file
submission = pd.DataFrame({'id': test_ids, 'target' : test_pred})
submission.to_csv('submission.csv', index=False)

# print the first 5 rows of the submission file
print(submission.head())

# print the directory where the submission file is saved
print(os.getcwd())

