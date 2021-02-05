import csv
from random import uniform

class Neuron:

    def __init__(self):
        self.weights = []
        self.net = 0
        self.error = 0

class NeuralNetwork:

    def __init__(self):
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = []
        self.data = []
        self.classes = set()

    def clear_net(self):
        for n in self.input_layer:
            n.net = 0
        for n in self.hidden_layer:
            n.net = 0
        for n in self.output_layer:
            n.net = 0

    def read_train_csv(self, csv_file, delimiter=','):
        try:
            with open(csv_file, newline='') as f:
                reader = csv.reader(f, delimiter=delimiter)
                first_line = reader.__next__()
                num_inputs = len(first_line) - 1
                print(f'Número de inputs: {num_inputs}')
                for row in reader:
                    self.data.append(([int(x) for x in row[:-1]], row[-1]))
                    self.classes.add(row[-1])
                num_outputs = len(self.classes)
                num_hidden = round((num_inputs * num_outputs) ** 0.5)
                print(f'Número de outputs: {num_outputs}')
                print(f'Número de neurônios na camada oculta: {num_hidden}')
                for _ in range(num_outputs):
                    neuron = Neuron()
                    self.output_layer.append(neuron)
                for _ in range(num_hidden):
                    neuron = Neuron()
                    for n in self.output_layer:
                        neuron.weights.append([n, round(uniform(-0.5, 0.5), 4)])
                    self.hidden_layer.append(neuron)
                for _ in range(num_inputs):
                    neuron = Neuron()
                    for n in self.hidden_layer:
                        neuron.weights.append([n, round(uniform(-0.5, 0.5), 4)])
                    self.input_layer.append(neuron)
        except FileNotFoundError:
            print(f'Arquivo com nome {csv_file} não encontrado.')

    def backpropagation(self, function, dx_function, learning_rate):
        cont = 0
        for row in self.data:
            cont += 1
            print(f'Iteracão: {cont}')
            data = row[0]
            label = row[1]
            for i in range(len(data)):
                self.input_layer[i].net = data[i]
            for n in self.input_layer:
                for w in n.weights:
                    w[0].net += n.net * w[1]
            for n in self.hidden_layer:
                for w in n.weights:
                    w[0].net += function(n.net) * w[1]
            for i in range(len(self.output_layer)):
                classes = list(self.classes)
                if classes[i] != label:
                    self.output_layer[i].error = (0 - function(self.output_layer[i].net)) * (dx_function(self.output_layer[i].net))
                else:
                    self.output_layer[i].error = (1 - function(self.output_layer[i].net)) * (dx_function(self.output_layer[i].net))
            for n in self.hidden_layer:
                s = 0
                for w in n.weights:
                    s += w[0].error * w[1]
                n.error = dx_function(n.net) * s
            for n in self.hidden_layer:
                for w in n.weights:
                    x = learning_rate * w[0].error * function(n.net)
                    w[1] += x
            for n in self.input_layer:
                for w in n.weights:
                    w[1] += learning_rate * w[0].error * n.net
            err = sum([n.error ** 2 for n in self.output_layer]) / 2
            print(f'Erro da rede: {err}')
            self.clear_net()
            

    def test(self, csv_file, function, delimiter=','):
        data = []
        try:
            with open(csv_file, newline='') as f:
                reader = csv.reader(f, delimiter=delimiter)
                reader.__next__()
                for row in reader:
                    data.append(([int(x) for x in row[:-1]], row[-1]))
        except FileNotFoundError:
            print(f'Arquivo com nome {csv_file} não encontrado.')
        classes = list(self.classes)
        for row in data:
            d = row[0]
            label = row[1]
            outputs = []
            for i in range(len(self.input_layer)):
                self.input_layer[i].net = d[i]
            for n in self.input_layer:
                for w in n.weights:
                    w[0].net += n.net * w[1]
            for n in self.hidden_layer:
                for w in n.weights:
                    w[0].net += function(n.net) * w[1]
            for n in self.output_layer:
                outputs.append(function(n.net))
            if outputs.index(max(outputs)) == classes.index(label):
                print('Correto!')
            else:
                print('Errado!')
            self.clear_net()