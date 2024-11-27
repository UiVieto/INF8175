import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset

class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.DotProduct:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        prediction = nn.as_scalar(self.run(x))

        return 1 if prediction >= 0 else -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        training_completed = False

        while not training_completed:
            training_completed = True
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)

                if prediction != nn.as_scalar(y):
                    training_completed = False
                    self.w.update(x, nn.as_scalar(y))


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        self.w1 = nn.Parameter(1, 200) 
        self.b1 = nn.Parameter(1, 200)

        self.w2 = nn.Parameter(200, 200)
        self.b2 = nn.Parameter(1, 200)

        self.w3 = nn.Parameter(200, 1)
        self.b3 = nn.Parameter(1, 1)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        # Première couche
        result = nn.Linear(x, self.w1)
        result = nn.AddBias(result, self.b1)
        
        # Deuxième couche
        result = nn.ReLU(result)
        result = nn.Linear(result, self.w2)
        result = nn.AddBias(result, self.b2)

        # Troisième couche
        result = nn.ReLU(result)
        result = nn.Linear(result, self.w3)
        result = nn.AddBias(result, self.b3)

        return result

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        predictions = self.run(x)

        return nn.SquareLoss(predictions, y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        learning_rate = 0.01
        total_loss = float('inf')
        while total_loss > 0.01:
            for x, y in dataset.iterate_once(10):
                loss = self.get_loss(x, y)

                gradient = nn.gradients(loss, [
                    self.w1, self.b1, 
                    self.w2, self.b2,
                    self.w3, self.b3,
                ])

                self.w1.update(gradient[0], -learning_rate)
                self.b1.update(gradient[1], -learning_rate)

                self.w2.update(gradient[2], -learning_rate)
                self.b2.update(gradient[3], -learning_rate)

                self.w3.update(gradient[4], -learning_rate)
                self.b3.update(gradient[5], -learning_rate)

            for x, y in dataset.iterate_once(dataset.y.size):
                total_loss = nn.as_scalar(self.get_loss(x, y))


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        self.w1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)

        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)

        self.w3 = nn.Parameter(128, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        result = nn.Linear(x, self.w1)
        result = nn.AddBias(result, self.b1)

        result = nn.ReLU(result)
        result = nn.Linear(result, self.w2)
        result = nn.AddBias(result, self.b2)

        result = nn.ReLU(result)
        result = nn.Linear(result, self.w3)
        result = nn.AddBias(result, self.b3)

        return result

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        validation_accuracy = 0.0
        learning_rate = 0.1

        while validation_accuracy < 0.97:
            for x, y in dataset.iterate_once(100):
                self.run(x)
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [
                    self.w1, self.b1,
                    self.w2, self.b2,
                    self.w3, self.b3,
                ])

                self.w1.update(gradients[0], -learning_rate)
                self.b1.update(gradients[1], -learning_rate)

                self.w2.update(gradients[2], -learning_rate)
                self.b2.update(gradients[3], -learning_rate)

                self.w3.update(gradients[4], -learning_rate)
                self.b3.update(gradients[5], -learning_rate)

                validation_accuracy = dataset.get_validation_accuracy()
