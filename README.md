
# NaiveBayes Module

This Python module provides three Naive Bayes classifiers - `GaussianNB`, `NaiveBayes`, and `CategoricalNB`. These classifiers are designed for classification tasks and offer different strategies for handling different types of data.

## GaussianNB

### Usage

```python
from naiveBayes import GaussianNB

gnb_classifier = GaussianNB()
X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = np.array([0, 1, 0])
gnb_classifier.fit(X_train, y_train)
predictions = gnb_classifier.predict(X_test)
```

### Methods

#### `fit(_X, y)`

Fits the model to the training data.

#### `predict(test)`

Predicts the class labels for the input test data.

## NaiveBayes

### Usage

```python
from naiveBayes import NaiveBayes

nb_classifier = NaiveBayes()
X_train = np.array([[1, 0], [1, 1], [0, 1]])
y_train = np.array([0, 1, 0])
nb_classifier.fit(X_train, y_train)
predictions = nb_classifier.predict(X_test)
```

### Methods

#### `fit(_X, y)`

Fits the model to the training data.

#### `predict(test)`

Predicts the class labels for the input test data.

## CategoricalNB

### Usage

```python
from naiveBayes import CategoricalNB

catnb_classifier = CategoricalNB()
X_train = pd.DataFrame({'feature1': ['a', 'b', 'a', 'b'], 'feature2': ['x', 'y', 'x', 'y']})
y_train = pd.Series([0, 1, 0, 1])
catnb_classifier.fit(X_train, y_train)
predictions = catnb_classifier.predict(X_test)
```

### Methods

#### `fit(X, y)`

Fits the model to the training data.

#### `predict(df)`

Predicts the class labels for the input DataFrame.

Feel free to modify and enhance this readme according to your preferences and any additional information you want to provide about the module.