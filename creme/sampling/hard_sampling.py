import collections

import random

from .. import base
from .. import optim
from .. import utils


class Triplet(collections.namedtuple('Triplet', 'x y loss')):

    def __lt__(self, other):
        return self.loss < other.loss


class HardSampling(base.Wrapper):
    """Hard sampler."""

    def __init__(self, model, loss_function, size, p,seed = None):
        self.model = model
        self.loss_function = loss_function
        self.pred_func = model.predict_one

        if isinstance(model, base.Classifier):
            self.pred_func = model.predict_proba_one

        elif isinstance(model, base.BinaryClassifier):
            self.pred_func = lambda x: model.predict_proba_one(x)[True]

        self.p = p
        self.size = size
        self.buffer = utils.SortedWindow(self.size)
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def _model(self):
        return self.model

    def predict_one(self, x):
        return self.model.predict_one(x)

    def fit_one(self, x, y):
        """
        Compute the loss of the input observation. If the loss is high enough,  the tuple (x, y)
        will be stored in the buffer. Update the model with observations stored in the buffer with a
        probability p. Update the model with input observation with a probability (1 - p). Compute
        the loss a second times if the model fitted an observation from the buffer.

        Parameters:
            x (Dict): Features.
            y (float): Target.

        """
        loss = self.loss_function.eval(y_true=y, y_pred=self.pred_func(x))

        if len(self.buffer) < self.size:
            self.buffer.append(Triplet(x=x, y=y, loss=loss))

        elif loss > self.buffer[0].loss:
            self.buffer.pop(0)

            self.buffer.append(Triplet(x=x, y=y, loss=loss))

        # Probability p
        if self._rng.uniform(0, 1) <= self.p:
            i = self._rng.randint(0, len(self.buffer) - 1)

            triplet = self.buffer.pop(i)

            self.model.fit_one(triplet.x, triplet.y)

            loss = self.loss_function.eval(y_true=triplet.y, y_pred=self.pred_func(triplet.x))

            self.buffer.append(Triplet(x=triplet.x, y=triplet.y, loss=loss))

        # Probability (1 - p).
        else:
            self.model.fit_one(x, y)

        return self


class HardSamplingRegressor(HardSampling):
    """Hard-sampling regressor.

    This wrapper enables a model to retrain on past samples who's output was hard to predict.
    This works by storing the hardest samples in a buffer of a fixed size. When a new sample
    arrives, the wrapped model is either trained on one of the buffered samples with a probability p
    or on the new sample with a probability (1 - p).

    The hardness of an observation is evaluated with a loss function that compares the sample's
    ground truth with the wrapped model's prediction. If the buffer is not full, then the sample
    is added to the buffer. If the buffer is full and the new sample has a bigger loss than the
    lowest loss in the buffer, then the sample takes it's place.

    Parameters:
        regressor (base.Regressor)
        loss_function (creme.optim.losses.RegressionLoss): Criterion used to evaluate the hardness of a sample.
        size (int): Size of the buffer.
        p (float): Probability of updating the model with a sample from the buffer instead of a
            new incoming sample.
        seed (int): Random seed.

    Example:

        ::

            >>> from creme import datasets
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import sampling

            >>> model = preprocessing.StandardScaler()

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     sampling.HardSamplingRegressor(
            ...         regressor = linear_model.LinearRegression(),
            ...          p = 0.2,
            ...          size = 30,
            ...         seed = 42,
            ...     )
            ... )

            >>> model_selection.progressive_val_score(
            ...     datasets.TrumpApproval(),
            ...     model,
            ...     metrics.MAE(),
            ...     print_every = 500
            ... )
            [500] MAE: 3.240558
            [1,000] MAE: 1.949106
            MAE: 1.947374

    """
    def __init__(self, regressor, size, p, loss_function = None, seed=None):
        if loss_function is None:
            loss_function =  optim.losses.Absolute()
        super().__init__(model=regressor, loss_function=loss_function, size=size,
            p=p, seed=seed
        )


class HardSamplingClassifier(HardSampling):
    """HardSamplingClassifier

    This wrapper enables a model to retrain on past samples who's output was hard to predict.
    This works by storing the hardest samples in a buffer of a fixed size. When a new sample
    arrives, the wrapped model is either trained on one of the buffered samples with a probability p
    or on the new sample with a probability (1 - p).

    The hardness of an observation is evaluated with a loss function that compares the sample's
    ground truth with the wrapped model's prediction. If the buffer is not full, then the sample
    is added to the buffer. If the buffer is full and the new sample has a bigger loss than the
    lowest loss in the buffer, then the sample takes it's place.

    Parameters:
        classifier (base.Classifier)
        loss_function (creme.optim.losses.CrossEntropy): Criterion used to evaluate the hardness
            of a sample.
        size (int): Size of the buffer.
        p (float): Probability of updating the model with a sample from the buffer instead of a
            new incoming sample.
        seed (int): Random seed.


    Example:

        ::

            >>> from creme import datasets
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import sampling

            >>> model = preprocessing.StandardScaler()

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     sampling.HardSamplingClassifier(
            ...         classifier = linear_model.LogisticRegression(),
            ...          p = 0.1,
            ...          size = 40,
            ...         seed = 42,
            ...     )
            ... )

            >>> model_selection.progressive_val_score(
            ...     X_y = datasets.Phishing(),
            ...     model = model,
            ...     metric = metrics.ROCAUC(),
            ...     print_every = 500,
            ... )
            [500] ROCAUC: 0.928271
            [1,000] ROCAUC: 0.948547
            ROCAUC: 0.952755

    """
    def __init__(self, classifier, size, p, loss_function=None, seed=None):
        if loss_function is None:
            loss_function =  optim.losses.CrossEntropy()

        HardSampling.__init__(self, model=classifier, loss_function=loss_function, size=size,
            p=p, seed=seed
        )

    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)