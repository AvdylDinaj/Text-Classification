import pandas as pd
import neattext.functions as nfx
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

#dataset load and inspect
df = pd.read_csv("C:/Users/Lenovo/Documents/Desktop/classification/netflix_users.csv")
print(df.shape)
print(df.info)
print(df.head())

#Preprocessing
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df = df.applymap(lambda x: nfx.remove_stopwords(nfx.remove_special_characters(str(x).lower()))
                 if isinstance(x, str) else x)

output_path = "C:/Users/Lenovo/Documents/Desktop/classification/netflix_users_clean.csv"
df.to_csv(output_path, index=False)


