import pandas as pd
import pickle
import joblib
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings('ignore')


filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)

#Input and Response
faq = pd.read_csv('FAQ.txt', delimiter='\t', encoding="utf-8", header = None)
faq = faq.set_index(0).to_dict()[1]

quit = "N"
while quit != "Y":
    question = input("Enter a question: ")
    category = loaded_model.predict([question])
    output = faq[category[0]]
    print(output)
    quit = input("Would you like to quit? (Y/N): ")
print("You have exited the chat")
