pip install VaderSentiment
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
df1 = pd.read_csv('dow_jones_news.csv')
df2 = pd.read_csv('dow_jones_stock.csv')
df1.head(5)
df1.shape
df2.head(5)
df2.shape
merge = df1.merge(df2, how='inner', left_index=True, right_index=True)
merge
merge.drop(columns=['Date_y'], inplace=True)
merge
headlines = []

for row in range(0, len(merge.index)):
  headlines.append(' '.join(str(x) for x in merge.iloc[row, 2:27]))
clean_headlines = []

for i in range(0, len(headlines)):
  clean_headlines.append(re.sub("b[(')]",'', headlines[i]))
  clean_headlines[i] = re.sub('b[(")]', '', clean_headlines[i])
  clean_headlines[i] = re.sub("\'", '', clean_headlines[i])
clean_headlines[20]
merge['combined_news'] = clean_headlines

merge['combined_news'][0]
merge.head(5)
def getsubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
  return TextBlob(text).sentiment.polarity
merge['subjective'] = merge['combined_news'].apply(getsubjectivity)
merge['Polarity'] = merge['combined_news'].apply(getPolarity)
merge.head(5)
def getSIA(text):
  sia = SentimentIntensityAnalyzer()
  sentiment = sia.polarity_scores(text)
  return sentiment
compound = []
neg = []
pos = []
neu = []
SIA = 0

for i in range(0, len(merge['combined_news'])):
  SIA = getSIA(merge['combined_news'][i])
  compound.append(SIA['compound'])
  neg.append(SIA['neg'])
  neu.append(SIA['neu'])
  pos.append(SIA['pos'])
merge['compound'] = compound
merge['negative'] = neg
merge['neutral'] = neu
merge['positive'] = pos
merge.head(5)
keep_columns = [ 'Open', 'High', 'Low', 'Volume', 'subjective', 'Polarity', 'compound', 'negative', 'neutral' ,'positive',  'Label' ]
df = merge[keep_columns]
df
X = df
X = np.array(X.drop(['Label'], 1))

y = np.array(df['Label'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32)
model = LinearDiscriminantAnalysis().fit(x_train,y_train)
pred = model.predict(x_test)
pred
y_test
print(classification_report(y_test, pred))
from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

app = Flask(__name__)

def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        merge = pd.read_csv('path_to_your_csv_file.csv')  # Adjust the path to your CSV file
        compound = []
        neg = []
        pos = []
        neu = []
        SIA = 0

        for i in range(0, len(merge['combined_news'])):
            SIA = getSIA(merge['combined_news'][i])
            compound.append(SIA['compound'])
            neg.append(SIA['neg'])
            neu.append(SIA['neu'])
            pos.append(SIA['pos'])

        merge['compound'] = compounda
        merge['negative'] = neg
        merge['neutral'] = neu
        merge['positive'] = pos

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

        model = LinearDiscriminantAnalysis().fit(x_train, y_train)

        pred = model.predict(x_test)

        classification_result = classification_report(y_test, pred)

        return render_template('result.html', classification_result=classification_result)

if __name__ == '__main__':
    app.run(debug=True)

