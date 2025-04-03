
# models
spacy gives us both nlp capabilities and access to pretrained models.

all the approaches can be implemented with these options:
- spacy's english packages - lightweight
- BERT models - heavier

### spacy's english packages
using something like spacy.load("en_core_web_sm"), we get CNN-based models
these are faster to train, lighter, and work nicely with scikit-learn where we can pick classifiers
example:
```
classifier = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=spacy_preprocessor)),
    ('classifier', LinearSVC())
])
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```
### spacy with bert
using spacy.load("en_core_web_trf"), we get transformer-based models
works right out of the box without having to pick a specific classifier

```
nlp.add_pipe("textcat")
nlp.get_pipe("textcat").add_label("POSITIVE")
nlp.get_pipe("textcat").add_label("NEGATIVE")

# After training (simplified), use it like this:
doc = nlp("I love this product!")
print(doc.cats)  # Will show prediction scores like {"POSITIVE": 0.92, "NEGATIVE": 0.08}
```

for validation, we'll always use 33-33-33 split

# approaches

## 1. three classes
true, false, irrelevant

## 2. two classes
irrelevant == false.

Here are a few options:

### 2.1 direct classification
just completely ignore the irrelevant class

### 2.2 different weighting
for example: dataset could be 70% false, 30% irrelevant.
two main hyperparameters to play with:
- the ratio of false vs irrelevant
- including more "falsy" data than "true" data, like 60% falsy, 40% true

### 2.3 two-layered approach
do a preliminary step to filter out only relevant entries.
might be good since we can use more data, but could perform worse if the first layer is not accurate

> different weighting between irrelevant and false is still apply here

#### 2.3.1 classical techniques, then model
use classical techniques such as bag-of-words / tf-idf

#### 2.3.2 two ML models
train one model to separate relevant from irrelevant.  
then train another to figure out true from false
