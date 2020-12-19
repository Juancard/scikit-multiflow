import copy as cp
import numpy as np

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.bayes import NaiveBayes


class DynamicWeightedMajorityMultiLabel(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Dynamic Weighted Majority ensemble classifier for MultiLabel data.
    """

    class WeightedExpert:
        """
        Wrapper that includes an estimator and its weight.

        Parameters
        ----------
        estimator: StreamModel or sklearn.BaseEstimator
            The estimator to wrap.
        weight: array of float
            The estimator's weight for each label.
        labels: int
            The number of labels.
        """

        def __init__(self, estimator, weight, labels=1):
            self.estimator = estimator
            self.weight = np.full(
                shape=labels, fill_value=weight, dtype=np.int)

    def __init__(self, labels, n_estimators=5, base_estimator=NaiveBayes(),
                 period=50, beta=0.5, theta=0.01):
        """
        Creates a new instance of DynamicWeightedMajorityClassifier.
        """
        super().__init__()

        self.n_estimators = n_estimators
        self.base_estimator = base_estimator

        self.beta = beta
        self.theta = theta
        self.period = period

        # Following attributes are set later
        self.epochs = None
        self.num_classes = None
        self.experts = None

        self.labels = labels

        self.reset()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fits the model on the supplied X and y matrices.

        Since it's an ensemble learner, if X and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        sample_weight: numpy.ndarray of shape
        (n_samples), optional (default=None)
        Samples weight. If not provided, uniform
        weights are assumed. Usage varies depending
        on the base estimator.

        Returns
        -------
        DynamicWeightedMajorityClassifier
            self
        """
        if (y.ndim == 1):
            y = np.array([y])
        for i in range(len(X)):
            self.fit_single_sample(
                X[i:i+1, :], y[i:i+1, :], classes, sample_weight
            )
        return self

    def predict(self, X):
        """ predict

        The predict function will take an average of the predictions of its
        learners, weighted by their respective weights, and return the most
        likely class.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        """
        preds = [np.multiply(np.array(exp.estimator.predict(X)), exp.weight)
                 for exp in self.experts]

        sum_weights = [sum(exp.weight[label] for exp in self.experts)
                       for label in range(self.labels)]
        instances = X.shape[0]
        aggregate = np.array([[
            np.sum(
                [np.array(p[i][label])
                 for p in preds] / sum_weights[label],
                axis=0
            ) for label in range(self.labels)
        ] for i in range(instances)])

        # Round to nearest int
        return (np.array(aggregate) + 0.5).astype(int)

    def predict_proba(self, X):
        raise NotImplementedError

    def fit_single_sample(self, X, y, classes=None, sample_weight=None):
        """
        Fits a single sample of shape `X.shape=(1, n_attributes)` and `y.shape=(1)`

        Aggregates all experts' predictions, diminishes weight of experts whose
        predictions were wrong, and may create or remove experts every _period_
        samples.

        Finally, trains each individual expert on the provided data.

        Train loop as described by Kolter and Maloof in the original paper.


        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Features matrix used for partially updating the model.

        y: Array-like
            An array-like of all the class labels for the samples in X.

        classes: list
            List of all existing classes. This is an optional parameter.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Applicability
            depends on the base estimator.

        """
        self.epochs += 1
        self.num_classes = max(
            len(classes) if classes is not None else 0,
            (int(np.max(y)) + 1), self.num_classes)
        predictions = [np.zeros((self.num_classes,))
                       for label in range(self.labels)]

        max_weight = [0] * self.labels
        weakest_expert_weight = [1] * self.labels
        weakest_expert_index = None

        # print("Epoch: ", self.epochs)
        for i, exp in enumerate(self.experts):
            y_hat = exp.estimator.predict(X).flatten()
            # print("Expert {}: \n\t{}\n\t{}".format(
            # i, y_hat, y[0]
            # ))
            for y_pred_idx, y_pred, in enumerate(y_hat):
                if (
                        int(y_hat[y_pred_idx]) != int(y[0][y_pred_idx])) and (
                    self.epochs % self.period == 0
                ):
                    exp.weight[y_pred_idx] *= self.beta
                    # print("Label {} - New weight: {} - Weights: {}".format(
                    # i,
                    # exp.weight[y_pred_idx],
                    # exp.weight
                    # ))

                predictions[y_pred_idx][int(
                    y_hat[y_pred_idx])] += exp.weight[y_pred_idx]
                max_weight[y_pred_idx] = max(
                    max_weight[y_pred_idx], exp.weight[y_pred_idx])

            if np.mean(exp.weight) < np.mean(weakest_expert_weight):
                weakest_expert_index = i
                weakest_expert_weight = exp.weight

        y_hat = np.array([np.argmax(predictions, axis=1)])
        if self.epochs % self.period == 0:
            # print("Epoch: {} - Experts weight".format(self.epochs))
            # for idx, e in enumerate(self.experts):
            # print(idx, ": ", e.weight)
            self._scale_weights(max_weight)
            self._remove_experts()
            if np.any(y_hat != y):
                if len(self.experts) == self.n_estimators:
                    # print("Removing expert {}: {}".format(
                    # weakest_expert_index, weakest_expert_weight))
                    self.experts.pop(weakest_expert_index)
                self.experts.append(self._construct_new_expert())

        # Train individual experts
        for exp in self.experts:
            exp.estimator.partial_fit(X, y, classes, sample_weight)

    def get_expert_predictions(self, X):
        """
        Returns predictions of each class for each expert.
        In shape: (n_experts, n_samples)
        """
        return [exp.estimator.predict(X) for exp in self.experts]

    def reset(self):
        """
        Reset this ensemble learner.
        """
        self.epochs = 0
        self.num_classes = 2    # Minimum of 2 classes
        self.experts = [
            self._construct_new_expert()
        ]

    def _scale_weights(self, max_weight):
        """
        Scales the experts' weights such that the max is 1.
        """
        scale_factor = np.divide(1, max_weight)
        for exp in self.experts:
            exp.weight = exp.weight * scale_factor

    def _remove_experts(self):
        """
        Removes all experts whose weight is lower than self.theta.
        """
        self.experts = [ex for ex in self.experts if np.mean(
            ex.weight) >= self.theta]

    def _construct_new_expert(self):
        """
        Constructs a new WeightedExpert from the provided base_estimator.
        """
        return self.WeightedExpert(
            cp.deepcopy(self.base_estimator), 1, self.labels)
