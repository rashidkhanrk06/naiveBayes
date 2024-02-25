import numpy as np 
import pandas as pd


class GaussianNB:
    """
    Gaussian Naive Bayes classifier for classification tasks.

    Attributes:
        classes (numpy.ndarray): Array containing unique classes in the training data.
        _prior (numpy.ndarray): Array containing prior probabilities for each class.
        _mean (numpy.ndarray): Array containing mean values for each feature in each class.
        _var (numpy.ndarray): Array containing variance values for each feature in each class.
        n_samples (int): Number of samples in the training data.
        alpha (float): Laplace smoothing parameter.

    Methods:
        __init__(alpha=1.0): Initializes an instance of the Gaussian_NB classifier.
        fit(_X, y): Fits the model to the training data.
        predict(test): Predicts the class labels for the input test data.
        _predict(x): Predicts the class label for a single data point.
        _pdf(c_idx, x): Computes the probability density function for a given class and data point.

    Example:
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 0])
        nb_classifier = Gaussian_NB()
        nb_classifier.fit(X_train, y_train)
        X_test = np.array([[1, 2], [2, 3]])
        predictions = nb_classifier.predict(X_test)
    """

    def __init__(self, alpha=1.0):
        """
        Initializes an instance of the Gaussian_NB classifier.

        Parameters:
            alpha (float): Laplace smoothing parameter.

        Attributes:
            classes (numpy.ndarray): Array containing unique classes in the training data.
            _prior (numpy.ndarray): Array containing prior probabilities for each class.
            _mean (numpy.ndarray): Array containing mean values for each feature in each class.
            _var (numpy.ndarray): Array containing variance values for each feature in each class.
            n_samples (int): Number of samples in the training data.
            alpha (float): Laplace smoothing parameter.
        """
        self.classes = None
        self._prior = None
        self._mean = None
        self._var = None
        self.n_samples = None
        self.alpha = alpha

    def fit(self, _X, y):
        """
        Fits the model to the training data.

        Parameters:
            _X (numpy.ndarray): Training data features.
            y (numpy.ndarray): Training data labels.

        Returns:
            None
        """
        n_sample, n_features = _X.shape
        self.n_samples = _X.shape[0]
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        self._prior = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = _X[y == c]
            self._mean[idx, :] = np.mean(X_c, axis=0)
            self._var[idx, :] = np.var(X_c, axis=0)
            self._prior[idx] = X_c.shape[0] / float(n_sample)

    def predict(self, test):
        """
        Predicts the class labels for the input test data.

        Parameters:
            test (numpy.ndarray): Test data features.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        y_predict = None
        if test.ndim==1:
            y_predict = self._predict(test)
        else:
             y_predict = [self._predict(x) for x in test]
        return np.array(y_predict)       

    def _predict(self, x):
        """
        Predicts the class label for a single data point.

        Parameters:
            x (numpy.ndarray): Data point features.

        Returns:
            int: Predicted class label.
        """
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self._prior[idx])
            likelihood = np.sum(np.log(self._pdf(idx, x)))

            posterior = likelihood + prior
            posteriors.append(posterior)
        return int(self.classes[np.argmax(posteriors)])

    def _pdf(self, c_idx, x):
        """
        Computes the probability density function for a given class and data point.

        Parameters:
            c_idx (int): Index of the class.
            x (numpy.ndarray): Data point features.

        Returns:
            numpy.ndarray: Probability density function values.
        """
        mu = self._mean[c_idx]
        var = self._var[c_idx]
        _var = var + self.alpha
        _x = np.exp(-((x - mu) ** 2) / (2 * _var)) / (np.sqrt(2 * np.pi * _var))

        return _x


class NaiveBayes:
    """
    Naive Bayes classifier for classification tasks.

    Attributes:
        classes (numpy.ndarray): Array containing unique classes in the training data.
        _prior (numpy.ndarray): Array containing prior probabilities for each class.
        _prob (numpy.ndarray): Array containing class conditional probabilities.
        n_samples (int): Number of samples in the training data.
        alpha (float): Laplace smoothing parameter.

    Methods:
        __init__(alpha=1.0): Initializes an instance of the NaiveBayes classifier.
        fit(_X, y): Fits the model to the training data.
        predict(test): Predicts the class labels for the input test data.
        _predict(x): Predicts the class label for a single data point.
        _cal_prob(c_idx, x): Calculates the conditional probability for a given class and data point.

    Example:
        X_train = np.array([[1, 0], [1, 1], [0, 1]])
        y_train = np.array([0, 1, 0])
        nb_classifier = NaiveBayes()
        nb_classifier.fit(X_train, y_train)
        X_test = np.array([[1, 0], [0, 1]])
        predictions = nb_classifier.predict(X_test)
    """

    def __init__(self, alpha=1.0):
        """
        Initializes an instance of the NaiveBayes classifier.

        Parameters:
            alpha (float): Laplace smoothing parameter.

        Attributes:
            classes (numpy.ndarray): Array containing unique classes in the training data.
            _prior (numpy.ndarray): Array containing prior probabilities for each class.
            _prob (numpy.ndarray): Array containing class conditional probabilities.
            n_samples (int): Number of samples in the training data.
            alpha (float): Laplace smoothing parameter.
        """
        self.classes = None
        self._prior = None
        self._prob = None
        self.alpha = alpha
        self.classes_sum = []

    def fit(self, _X, y):
        """
        Fits the model to the training data.

        Parameters:
            _X (numpy.ndarray): Training data features.
            y (numpy.ndarray): Training data labels.

        Returns:
            None
        """
        n_sample, n_features = _X.shape
        self.n_samples = _X.shape[0]
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self._prob = np.zeros((n_classes, n_features))
        self._prior = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = _X[y == c]
            self._prob[idx, :] = np.sum(X_c, axis=0) 
            self._prior[idx] = X_c.shape[0] / n_sample
            self.classes_sum.append(np.sum(self._prob))

    def predict(self, test):
        """
        Predicts the class labels for the input test data.

        Parameters:
            test (numpy.ndarray): Test data features.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        y_predict = None
        if test.ndim==1:
            y_predict = self._predict(test)
        else:
             y_predict = [self._predict(x) for x in test]
        return np.array(y_predict)

    def _predict(self, x):
        """
        Predicts the class label for a single data point.

        Parameters:
            x (numpy.ndarray): Data point features.

        Returns:
            int: Predicted class label.
        """
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = self._prior[idx]
            likelihood = np.prod(self._cal_prob(idx, x))
            posterior = likelihood * prior
            posteriors.append(posterior)
        return int(self.classes[np.argmax(posteriors)])

    def _cal_prob(self, c_idx, x):
        """
        Calculates the conditional probability for a given class and data point.

        Parameters:
            c_idx (int): Index of the class.
            x (numpy.ndarray): Data point features.

        Returns:
            numpy.ndarray: Conditional probability values.
        """
        arr_ = self._prob[c_idx]
        indices = np.where(x > 0)
        _x = (arr_[indices]+self.alpha)/(self.classes_sum[c_idx]+(self.alpha*sum(self.classes_sum)))

        return _x



class CategoricalNB:
    """
    Categorical Naive Bayes classifier for classification tasks.

    Attributes:
        alpha (float): Laplace smoothing parameter.
        features (list): List of feature names.
        likelihoods (dict): Dictionary containing class-wise likelihoods for each feature.
        class_priors (dict): Dictionary containing class priors.
        pred_priors (dict): Dictionary containing priors for each unique value in features.
        y (pd.Series): Series containing target labels.
        X (pd.DataFrame): DataFrame containing feature values.

    Methods:
        __init__(alpha=1.0): Initializes an instance of the CategoricalNB classifier.
        fit(X, y): Fits the model to the training data.
        perior(): Computes and stores class priors.
        liklihood(): Computes and stores class-wise likelihoods for each feature.
        evidence(): Computes and stores priors for each unique value in features.
        predict(df): Predicts the class labels for the input DataFrame.

    Example:
        X_train = pd.DataFrame({'feature1': ['a', 'b', 'a', 'b'], 'feature2': ['x', 'y', 'x', 'y']})
        y_train = pd.Series([0, 1, 0, 1])
        nb_classifier = CategoricalNB()
        nb_classifier.fit(X_train, y_train)
        X_test = pd.DataFrame({'feature1': ['a', 'b'], 'feature2': ['x', 'y']})
        predictions = nb_classifier.predict(X_test)
    """

    def __init__(self, alpha=1.0):
        """
        Initializes an instance of the CategoricalNB classifier.

        Parameters:
            alpha (float): Laplace smoothing parameter.

        Attributes:
            alpha (float): Laplace smoothing parameter.
            features (list): List of feature names.
            likelihoods (dict): Dictionary containing class-wise likelihoods for each feature.
            class_priors (dict): Dictionary containing class priors.
            pred_priors (dict): Dictionary containing priors for each unique value in features.
            y (pd.Series): Series containing target labels.
            X (pd.DataFrame): DataFrame containing feature values.
        """
        self.alpha = alpha
        self.features = list
        self.likelihoods = {}
        self.class_priors = {}
        self.pred_priors = {}

        self.y = pd.Series
        self.X = pd.DataFrame

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the model to the training data.

        Parameters:
            X (pd.DataFrame): Training data features.
            y (pd.Series): Training data labels.

        Returns:
            None
        """
        self.features = X.columns
        self.X = X
        self.y = y
        self.perior()
        self.liklihood()
        self.evidence()

    def perior(self) -> dict:
        """
        Computes and stores class priors.

        Returns:
            None
        """
        classes, counts = np.unique(self.y, return_counts=True)
        for key, value in zip(classes, counts / len(self.y)):
            self.class_priors[key] = value

    def liklihood(self):
        """
        Computes and stores class-wise likelihoods for each feature.

        Returns:
            None
        """
        n_sample = self.X.shape[0]
        for c in self.class_priors:
            sub_classes = {}
            for x in self.X.columns:
                unique_classes = self.X[x].unique().tolist()

                x_c = self.X[x][self.y == c]
                outcomes = len(x_c)
                _class, counts = np.unique(x_c, return_counts=True)
                _dict = dict(zip(_class, (counts + self.alpha) / (outcomes + self.alpha / n_sample)))

                update_classes = {value: _dict[value] if value in _dict else 0 for value in unique_classes}

                sub_classes[x] = update_classes
            self.likelihoods[c] = sub_classes

    def evidence(self):
        """
        Computes and stores priors for each unique value in features.

        Returns:
            None
        """
        for i in self.X.columns:
            classes, counts = np.unique(self.X[i], return_counts=True)
            for key, value in zip(classes, counts):
                self.pred_priors[key] = value / self.X.shape[0]

    def predict(self, df: pd.DataFrame):
        """
        Predicts the class labels for the input DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing test data features.

        Returns:
            list: List of predicted class labels.
        """
        m = 3
        p = 1 / m
        result_lst = []
        for _, i in df.iterrows():
            result = {}
            for _is in self.class_priors:
                _likelihood = 1
                _evidence = 1

                for feature, row in zip(self.features, i.to_list()):
                    _likelihood *= self.likelihoods[_is][feature][row]
                    _evidence *= self.pred_priors[row]
                posterior = (_likelihood * self.class_priors[_is]) / (_evidence)
                result[_is] = posterior

            result_lst.append(max(result, key=result.get))
        return result_lst
