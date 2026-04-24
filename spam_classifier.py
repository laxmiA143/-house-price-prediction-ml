from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
emails = [
    "Win money now",
    "Hello, how are you?",
    "Claim your prize",
    "Meeting at 5pm",
    "Earn cash fast"
]

labels = [1, 0, 1, 0, 1]  # 1 = Spam, 0 = Not Spam

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Model
model = MultinomialNB()
model.fit(X, labels)

# Test
test_email = ["Win cash prize"]
test_vector = vectorizer.transform(test_email)

prediction = model.predict(test_vector)

print("Spam" if prediction[0] == 1 else "Not Spam")
