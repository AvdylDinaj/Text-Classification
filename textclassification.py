import pandas as pd
import neattext.functions as nfx
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix

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

def predict_genre_user_input():
    print("\nPlease enter the following information:")

    age = int(input("• Age: "))
    country = input("• Country: ").strip().title()
    subscription_type = input("• Subscription Type (Basic/Standard/Premium): ").strip().title()
    watch_time_hours = float(input("• Average Watch Time Hours per Month: "))
    input_df = pd.DataFrame([{
        "Age": age,
        "Country": country,
        "Subscription_Type": subscription_type,
        "Watch_Time_Hours": watch_time_hours
    }])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    try:
        pred_idx = log_reg.predict(input_df)[0]
        predicted_genre = le.inverse_transform([pred_idx])[0]
        probabilities = log_reg.predict_proba(input_df)[0]
        print("\n PREDICTION RESULTS")
        print(f"Predicted Favorite Genre: {predicted_genre}")

        print(f"\n Confidence Scores:")
        for i, genre in enumerate(le.classes_):
            confidence = probabilities[i] * 100
            print(f"   {genre}: {confidence:.1f}%")

    except Exception as e:
        print(f"Prediction error: {e}")
        return None

    return predicted_genre

if __name__ == "__main__":
    predicted_genre = predict_genre_user_input()

cm_lr = confusion_matrix(y_test, y_pred_lr)

#Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_lr.png')  # Save
print("Confusion matrix saved as 'confusion_matrix_lr.png'")

# Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)

#Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Reds',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_nb.png')  # Save
print("Confusion matrix saved as 'confusion_matrix_nb.png'")