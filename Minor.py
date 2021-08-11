import tkinter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
import tkinter.font
import tkinter.messagebox
from PIL import ImageTk,Image  
#root = Tk()  
#canvas = Canvas(root, width = 300, height = 300)  
#canvas.pack()  



window = tkinter.Tk()
window.title("Minor Project Group 4")
window.configure(bg = '#CCE1F2')
canvas = Canvas(window, width = 300, height = 300, bg = "#CCE1F2", borderwidth = 0 , highlightthickness = 0 )  
canvas.pack()    
img = ImageTk.PhotoImage(Image.open("jims.jpg"))  
canvas.create_image(20, 20, anchor=NW, image=img)

tkinter.Label(window, text = "Sentiment Analyser for Social Media", font = ("Ariel Black",50), bg = '#CCE1F2' ).pack()
# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) 


# Cleaning the texts
import re
from emot.emo_unicode import EMOTICONS
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)


# Creating the Bag of Words modelgooo
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

k = (accuracy_score(y_test, y_pred))
k = k*100
def clicked():
 tkinter.messagebox.showinfo('Accuracy', k)
 #tkinter.Label(window, text = k,font = ("Ariel Black",30)).pack()
#window.geometry('800*400') 
tkinter.Button (window, text= "Click for model accuracy",font = ("Ariel Black",40),bg = 'black',fg = "white", command = clicked).pack() 

def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', " ".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

tkinter.Label(window, text = "Enter the review/comment: ",font = ("Ariel Black",30), bg = '#CCE1F2').pack()
#text=input("Enter the review/comment: ")
text = tkinter.Entry(window, width=50, bd = 3, font = ("Ariel Black",30), bg = "black",fg = "white").pack()
text = str(text)
text = cv.transform([text]).toarray()
#text = tfidfconverter.transform(text).toarray()
label = classifier.predict(text)[0]

def work():
 
 
 if (label == 1):
  tkinter.Label(window, text = "Person didnt liked the food",font = ("Ariel Black",20), bg = '#CCE1F2').pack()
 else:
  tkinter.Label(window, text = "Likes the food",font = ("Ariel Black",20), bg = '#CCE1F2').pack()

#text = "Great service, staff and well baked prepared pizza"

      
tkinter.Button(window, text = "Enter" , command = work).pack()
#tkinter.Label(window, text = text,font = ("Ariel Black",30)).pack()
#text = "Great service, staff and well baked prepared pizza"

window.mainloop()