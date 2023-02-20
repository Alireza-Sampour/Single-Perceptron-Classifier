""" 
Implementation of a single layer of perceptron and visualize training process
"""


from random import shuffle, normalvariate
import matplotlib.pyplot as plt
from matplotlib.pyplot import axes


class Perceptron:
    
    def __init__(self, number_of_inputs: int, learning_rate: float = 0.5) -> None:
        """
        Constructor for the Perceptron class.
        
        Args:
            number_of_inputs (int): The number of input features (excluding the bias term) that the perceptron should expect.
            learning_rate (float): optional (default=0.5) The learning rate for the perceptron
                        which controls the step size in weight updates. 

        Returns:
            None

        The algorithm may diverge with a high learning rate and may converge too slowly with a low learning rate.
        """

        self.weights = [normalvariate(+1, sigma) for _ in range(number_of_inputs)]
        self.weights.insert(0, 1) # insert bias
        self.learning_rate = learning_rate
    
    
    def predict(self, inputs: list):
        """
        Predict function that takes in inputs and uses the weights to predict the class of the input
        
        Args:
            inputs (list): A list of inputs to be used for prediction

        Returns:
            int: Returns +1 if the activation is greater than 0 and -1 if it is equal or less than 0
        """

        activation = self.weights[0] + sum(w * x for w, x in zip(self.weights[1:], inputs))
        return +1 if activation >= 0 else -1 # Using the signum function to determine the output


    def train(self, training_data: list, number_of_epochs: int) -> None:
        """
        Trains the perceptron model on the given training data.
        
        Args:
            training_data (list): A list of tuples (input, target) representing the training data. 
                            'input' is a list of input features and 'target' is the expected target output.
            number_of_epochs (int): The maximum number of epochs to run.

        Returns:
            None
        """

        plt.ion()
        fig, ax = plt.subplots()
        fig.suptitle("Perceptron Classifier")
        desicion_boundaries = []

        for epoch in range(1, (number_of_epochs) + 1):
            number_of_errors = 0
            predictions = []
            actuals = []
            
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                
                predictions.append(prediction)
                actuals.append(target)
                
                if error != 0: # if error is not zero then update the weights
                    number_of_errors += 1
                    self.weights[0] += self.learning_rate * error # Loss
                    self.weights[1:] = [w + self.learning_rate * error * x for w, x in zip(self.weights[1:], inputs)]

            desicion_boundaries.append((self.weights.copy(), number_of_errors, self.calc_accuracy(predictions, actuals)))
            
            if number_of_errors == 0:
                print(f"Converged in {epoch} epochs")
                break
            
            shuffle(training_data)
            
            if epoch % 100 == 0:
                self.learning_rate = 1 / epoch
        
        else:
            print(f"Did not converge in {epoch} epochs")
            print(f"Best accuracy: {max(desicion_boundaries, key=lambda x: x[2])[2]*100}%")
        
        for weights, number_of_errors, accuracy in desicion_boundaries:
            self.plot_decision_boundary(training_data, ax, number_of_errors, weights, accuracy)
            plt.draw()
            plt.pause(0.5 if epoch != number_of_epochs else 0.001)

        plt.show(block=True)


    def plot_decision_boundary(self, training_data: list, ax: axes, err: int, weights: list, accuracy: float) -> None:
        """
        Plots the decision boundary for the given training data and weights.
        
        Args:
            training_data (list): A list of input, output pairs.
            ax (Axes): The axes on which to plot.
            err (int): The number of misclassified samples.
            weights (list): The weights used to calculate the decision boundary.
            accuracy (float): The accuracy of trained model.

        Returns:
            None
        """

        ax.clear()
        colors = {+1: 'red', -1: 'blue'}
        
        for inputs, label in training_data:
            ax.scatter(inputs[0], inputs[1], color=colors[label], edgecolors='black')
        
        """
        y = mx + c
        
        m: Slope
        c: The height at which the line crosses the y-axis (y-intercept) 
        
        m = -w1 / w2
        c = -b / w2
        
        So `y` can be written as:
        y = (-w1 / w2) * x + -b / w2
        """

        m = lambda w1, w2 : (-1 * w1) / w2
        c = lambda b, w2: (-1 * b) / w2

        x = [inputs[0] for inputs, _ in training_data]
        y = [(m(weights[1], weights[2]) * xi) + c(weights[0], weights[2]) for xi in x]
        
        ax.set_ylim(-5 , +5)
        ax.set_xlim(-5 , +5)
        ax.plot(x, y, 'g-', label = 'Decision boundary')
        ax.set_xlabel(f'misclassified items is {err} and accuracy is {accuracy * 100}%', labelpad = 7)
        ax.legend(loc = 'lower left')


    def generate_linearly_separable_data(self, number_of_samples: int) -> list:
        """
        Generate linearly separable data for binary classification

        Args:
            number_of_samples (int): number of samples to be generated

        Returns:
            list: A list of tuples containing input features and output labels
                    The input feature is a list of two values, x and y
                    The output label is -1 or +1
        """

        samples = []
        for i in range(number_of_samples):
            x = normalvariate((-1 if i % 2 == 0 else +1) * 1, sigma)
            y = normalvariate((-1 if i % 2 == 0 else +1) * 1, sigma)
            samples.append(([x, y], (-1 if i % 2 == 0 else +1)))
   
        return samples


    def calc_accuracy(self, prediction: int, actual: int):
        """
        Computes the accuracy score.
        
        Args:
            prediction (list): Predicted labels.
            actual (list): Actual labels.
        
        Returns:
            float: The accuracy score.
        """
        
        return sum(pred == act for pred, act in zip(prediction, actual)) / len(prediction)


if __name__ == '__main__':
    number_of_samples = 100
    sigma = 0.7
    perceptron = Perceptron(number_of_inputs = 2) # Create a perceptron network with 2 input
    perceptron.train(perceptron.generate_linearly_separable_data(number_of_samples), number_of_epochs=number_of_samples * number_of_samples)
