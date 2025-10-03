import pandas as pd
import neattext.functions as nfx
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

#dataset load and inspect
df = pd.read_csv("C:/Users/Lenovo/Documents/Desktop/classification/netflix_users.csv")
print(df.shape)
print(df.info)
print(df.head())

#Preprocessing
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df = df.applymap(lambda x: nfx.remove_stopwords(nfx.remove_special_characters(str(x).lower()))
                 if isinstance(x, str) else x)

#Exploratory Analysis
print(df['Subscription_Type'].value_counts())
print(df['Favorite_Genre'].value_counts())
print("The average of watched hours is:")
print(df['Watch_Time_Hours'].mean())

#Prepare data for training
X = df.drop('Subscription_Type', axis=1)
y = df['Favorite_Genre']
le = LabelEncoder()
y = le.fit_transform(y)
X = pd.get_dummies(X)

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

#TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"TF-IDF features shape - Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")



#Logisitic regression

log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
