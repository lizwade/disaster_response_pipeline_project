import json
import plotly
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from PIL import Image

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/liz_clean.db')
df = pd.read_sql_table('disaster', engine)

# load model
model = joblib.load("../models/trained_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    label_counts = df.iloc[:,4:].sum()

    #functions and code to extract data for WordCloud
    def create_bow(a_series):
        bow = ""
        for message in a_series:
            bow = bow + message
        return bow

    #I got help from https://www.datacamp.com/community/tutorials/wordcloud-python
    def display_wordcloud(label, df=df):
        '''
        Creates a word cloud using messages tagged with the given label
        '''
        wordcloud = WordCloud().generate(create_bow(df[df[label]==1]['message']))
        #plt.imshow(wordcloud, interpolation='bilinear')
        #plt.axis("off")
        #plt.show()
        return wordcloud

    shelter = display_wordcloud('shelter')
    weather = display_wordcloud('weather_related')

    # create visuals
    # Sadly I could not work out how to render the wordcloud in html
    # So I instead extract and plot the words and frequencies
    # from the wordcloud objects
    graphs = [
        {
            'data': [
                Bar(
                    x=list(label_counts.index),
                    y=label_counts.values

                )
            ],

            'layout': {
                'title': 'Distribution of labels',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label"
                },
                'barmode': 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=list(shelter.words_.keys()),
                    y=list(shelter.words_.values()),
                    #orientation = 'h'

                )
            ],

            'layout': {
                'title': 'Top words in SHELTER messages',
                'yaxis': {
                    'title': "Relative frequency"
                },
                'xaxis': {
                    'title': "Word"
                },
                'barmode': 'group'

            }


        },
        {
            'data': [
                Bar(
                    x=list(weather.words_.keys()),
                    y=list(weather.words_.values()),
                    #orientation = 'h'

                )
            ],

            'layout': {
                'title': 'Top words in WEATHER_RELATED messages',
                'yaxis': {
                    'title': "Relative frequency"
                },
                'xaxis': {
                    'title': "Word"
                },
                'barmode': 'group'

            }


        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
