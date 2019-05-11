import sys
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
import pickle

def load_data(database_filepath):
    '''
    Reads the sql table 'disaster' from the given filepath
    and returns the feature X, the targets Y and category_names

    Parameters:
    database_filepath (string): the filepath to an sql table 'disaster'

    Returns:
    X (Series of strings): the messages
    Y (DataFrame): 1 in columns where category applies to message, otherwise 0
    category_names (List of strings): Label for each category

    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster', engine)

    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    '''
    takes a string and returns a list of cleaned tokens

    Parameters:
    text (string): the text to be tokenized

    Returns:
    List of strings: tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Constructs and returns a classifier pipeline

    Parameters:
    None

    Returns:
    Pipeline: classifier pipeline
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {
              'vect__max_features': [None, 100, 1000],
              'vect__max_df': [0.5, 1.0, 2.0],
              'vect__ngram_range': [(1, 1), (1,2)],
              'moc__estimator__min_samples_leaf': [1,3],
              'moc__estimator__n_estimators': [10, 50, 100],
              'moc__estimator__min_samples_split': [2,3,4]
             }


    cv = GridSearchCV(pipeline, parameters)

    return cv
    '''
    Note that with these grid parameters, this will take many hours to train.
    Previous training on the given data sets found the following best params:
    {'moc__estimator__min_samples_leaf': 1,
    'moc__estimator__min_samples_split': 2,
    'moc__estimator__n_estimators': 100,
    'vect__max_df': 1.0,
    'vect__max_features': 1000,
    'vect__ngram_range': (1, 2)}
    '''



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints a classification report for each category

    Parameters:
    model (Pipeline): the model to evaluate
    X_test (Series of strings): messages to be classified
    Y_test (Series of int): true classifications
    category_names: names for categories
    '''
    Y_pred = model.predict(X_test)

    for col in range(36):
        pred = Y_pred[:,col]
        true = np.array(Y_test.iloc[:,col])
        print("Category:",category_names[col])
        print(classification_report(pred,true))


def save_model(model, model_filepath):
    outfile = open('trained_model.pkl','wb')
    pickle.dump(model, outfile)
    outfile.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
