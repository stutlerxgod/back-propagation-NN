import numpy as np


class NeuralNetwork(object):

    def __init__(self, num_inputs, hidden_layers, neurons_in_hidden_layer, num_outputs):

        self.num_inputs = num_inputs
        self.hidden_layers = [neurons_in_hidden_layer for i in range(hidden_layers)]  # usefully when hidden_layers > 1
        self.num_outputs = num_outputs

        layers = [num_inputs] + self.hidden_layers + [num_outputs]


        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights


        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives


        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations


    def forward_prop(self, inputs):
        activations = inputs
        self.activations[0] = activations

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)

            activations = self.sigmoid(net_inputs)
            self.activations[i + 1] = activations

        return activations


    def back_prop(self, error):

        for i in reversed(range(len(self.derivatives))):

            activations = self.activations[i+1]

            delta = error * self.sigmoid(activations, derivation=True)  # [...]

            delta_re = delta.reshape(delta.shape[0], -1).T  # [[...]]

            current_activations = self.activations[i]  # 1x3 matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)  # 3x1 matrix

            self.derivatives[i] = np.dot(current_activations, delta_re)

            error = np.dot(delta, self.weights[i].T)


    def train(self, inputs, targets, epochs=50, learning_rate=0.1):

        for i in range(epochs):
            sum_errors = 0

            for j, input in enumerate(inputs):
                target = targets[j]

                output = self.forward_prop(input)

                error = target - output

                self.back_prop(error)

                self.gradient_descent(learning_rate)

                sum_errors += self.mse(target, output)

            # Report an error after each epoch
            print("Epoch {}. Error: {}".format(i+1, (sum_errors / len(inputs))))

        print("Training complete!")


    def gradient_descent(self, learningRate):

        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate


    def sigmoid(self, v, derivation=False):
        if derivation:
            return v * (1.0 - v)
        return 1.0 / (1 + np.exp(-v))


    def mse(self, target, output):
        return np.average((target - output) ** 2)


def main():
    # Creating simple Dataset that contains sum of 2 numbers -- First dataset option
    import random
    items = np.array([[random.random()/2 for i in range(2)] for j in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])


    # # Add iris Dataset -- Second dataset option
    # from sklearn import datasets
    # iris = datasets.load_iris()
    # items=np.array(iris['data'][:100])
    # targets=np.array([[i] for i in iris['target'][:100]])


    # Creating NeuralNetwork Class
    nn = NeuralNetwork(num_inputs=len(items[0]), hidden_layers=1, neurons_in_hidden_layer=3, num_outputs=len(targets[0]))

    # Training Neural Network
    nn.train(items, targets, epochs=50, learning_rate=0.7)


    # Prediction for sum of nums dataset
    print("\nPrediction:")
    test_input = np.array([0.5, 0.2])  # Target is to get number as closer to 0.7
    test_output = nn.forward_prop(test_input)
    print("Network believes that {} + {} = {}".format(test_input[0], test_input[1], test_output[0]))


    # # Prediction for iris dataset
    # test_input = np.array([5.1, 3.5, 1.4, 0.2])  # Target is to get number as closer to 0
    # test_output = nn.forward_prop(test_input)
    # print("Network believes that {} is equal to {}".format(test_input, test_output[0]))
    #
    # test_input = np.array([7.0, 3.2, 4.7, 1.4])  # Target is to get number as closer to 1
    # test_output = nn.forward_prop(test_input)
    # print("Network believes that {} is equal to {}".format(test_input, test_output[0]))


if __name__ == "__main__":
    main()
