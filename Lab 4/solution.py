# IMPORTS
import numpy as np
import re
import heapq
import copy
import sys

# GLOBALS
# mapa funkcija na simbol iz konfiguracije mreže po OO tvornica
activation_factory = dict()

# UTIL
# pretvara ulaz u tuple (ulazi, izlazi)
def parse_to_datapoint(raw : str):
    data = [float(x) for x in raw.split(",")]
    return tuple([data[:-1], [data[-1]]])

# čita file i parsira ulaze/izlaze
def generate_data(path : str):
    dataset, varnum = [], None
    with open(path) as f:
        varnum = len(f.readline().strip().split(","))
        for line in f:
            dataset.append(parse_to_datapoint(line.strip()))
    return dataset, varnum

# roulette wheel odabri count elemenata s težinama
def pick_n(elements : list, probabilities : list, count : int):
    selected = []
    prob_sum = sum(probabilities)
    while(count):
        threshold, accu, it = np.random.rand() * prob_sum, 0., 0
        while(1):
            accu += probabilities[it]
            if(accu > threshold):
                break
            it += 1
        selected.append(elements[it])
        count -= 1
    return tuple(selected)

# NEURAL NET UTIL
# implementacija sigmoide
def sigmoid(x : float):
    return 1./(1. + np.exp(-x))

# nasumično generiranje matrice zadanih dimenzija
def random_weights(width : int, height : int):
    return np.random.normal(0.0, 0.01, size=(width, height))

# implementacija srednje kvadratne greške
def MSE(gotten : list, expected : list):
    N, s = len(gotten), 0.
    for i in range(N):
        s += (gotten[i] - expected[i])**2
    return s / N

# LAYER
# implementacija jednog sloja mreže po OO strategija
class layer:
    # konstruktor
    def __init__(self, activation : callable, node_count : int, prev_layer_count : int):
        self.activation = activation
        self.weights = random_weights(node_count, prev_layer_count)
        self.bias = random_weights(node_count, 1).flatten()

    # vrijednost prolaza kroz sloj za dane ulaze
    def evaluate(self, inputs : list):
        return self.activation(self.weights @ inputs + self.bias)

# NEURAL NET
# implementacija neuronske mreže po OO strategija i OO kompozit
class nerual_network:
    # konstruktor
    def __init__(self, config : str, node_count : int, loss : callable):
        self.layers = []
        self.last_output = None
        self.loss = loss
        configs = re.findall("[0-9]+[a-zA-Z]+", config)
        prev = node_count
        for i in configs:
            subconfig = re.match("([0-9]+)([a-zA-Z]+)$", i)
            self.layers.append(layer(activation_factory[subconfig.group(2)], int(subconfig.group(1)), prev))
            prev = int(subconfig.group(1))
        self.layers.append(layer(lambda x : x, 1, prev))
   
    # evaluacija rezultata mreže za dane ulaze
    def evaluate(self, inputs : list):
        for i in range(len(self.layers)):
            inputs = self.layers[i].evaluate(inputs)

        self.last_output = inputs.copy()
        return inputs
    
    # funkcija za određivanje greške/gubitka za dani ulaz
    def train(self, data : list):
        inputs = [x[0] for x in data]
        expected = [x[1] for x in data]

        err = 0.
        for i in range(len(data)):
            gotten = self.evaluate(inputs[i])
            err += self.loss(gotten, expected[i])
   
        return err / len(data)

# GENETIC UTIL
# funkcija za stvaranje djeteta, po pravilu da su djetetove težine prosjek roditeljevih
def crossover_average(nn1 : nerual_network, nn2 : nerual_network):
    new_nn = copy.deepcopy(nn1)
    for i in range(len(nn1.layers)):
        new_nn.layers[i].weights = (nn1.layers[i].weights + nn2.layers[i].weights) / 2
        new_nn.layers[i].bias = (nn1.layers[i].bias + nn2.layers[i].bias) / 2
    return new_nn

# funkcija koja nasumično mutira svaku težinu po normalnoj distribuciji
def mutate_gauss_normal(nn : nerual_network, mut_probability : float, mut_scale : float):
    for layer in nn.layers:
        for i in range(layer.weights.shape[0]):
            for j in range(layer.weights.shape[1]):
                if(np.random.rand() < mut_probability):
                    layer.weights[i, j] += np.random.normal(0, mut_scale)

            if(np.random.rand() < mut_probability):
                layer.bias[i] += np.random.normal(0, mut_scale)

# GENETIC
# implementacija genetskog algoritma po OO okvirna metoda i OO strategija 
class genetic_algorithm:
    # konstruktor
    def __init__(self, test_set : list, crossover : callable, mutate : callable, config : str, input_num : int, loss : callable, popsize : int, elitism : int, mut_probability : float, mut_scale : float, iter_count : int):
        self.test_set = test_set
        self.crossover = crossover
        self.mutate = mutate
       
        self.popsize = popsize
        self.elitism = elitism
        self.mut_probability = mut_probability
        self.mut_scale = mut_scale
        self.iter_count = iter_count

        self.population = [nerual_network(config, input_num, loss) for _ in range(self.popsize)]

    # okvirna metoda za treniranje, koristi heapq za brzo sortiranje elemenata, fitness računa kao recipročna vrijednost pogreške
    def train(self, data : list):
        ranked = None
        for it in range(1, self.iter_count + 1):
            ranked = []
            for i in range(self.popsize):
                err = self.population[i].train(data)
                heapq.heappush(ranked, (-1./err, i, self.population[i]))
           
            if(not(it % 2000)):
                print(f"[Train error @{it}]: {self.test(ranked[0][2], data):.6f}")

            # elitizam
            new_pop = [copy.deepcopy(ranked[i][2]) for i in range(self.elitism)]
           
            fits = [-x for x, _, _ in ranked]
            fits_accu = sum(fits)
            probs = [fit/fits_accu for fit in fits]

            self.population = [nn for _, _, nn in ranked]
            while(len(new_pop) < self.popsize):
                nn1, nn2 = pick_n(self.population, probs, 2)                # selekcija
                new_nn = self.crossover(nn1, nn2)                           # križanje
                self.mutate(new_nn, self.mut_probability, self.mut_scale)   # mutacija
                new_pop.append(new_nn)
           
            self.population = new_pop

        print(f"[Test error]: {self.test(ranked[0][2]):.6f}")
    
    # testiranje funkcije na danom ili testnom skupu
    def test(self, best : nerual_network, data : list = None):
        if(data):
            return best.train(data)
        return best.train(test)

# Dodavanje sigmoide u tvornicu aktivacija
activation_factory.update({"s" : sigmoid})

# MAIN
if __name__ == "__main__":
    popsize, elitism, mut_probability, mut_scale, iter_count, train_path, test_path, config = None, None, None, None, None, None, None, None

    try:
        popsize = int(sys.argv[sys.argv.index("--popsize") + 1])
        elitism = int(sys.argv[sys.argv.index("--elitism") + 1])
        mut_probability = float(sys.argv[sys.argv.index("--p") + 1])
        mut_scale = float(sys.argv[sys.argv.index("--K") + 1])
        iter_count = int(sys.argv[sys.argv.index("--iter") + 1])
        train_path = sys.argv[sys.argv.index("--train") + 1]
        test_path = sys.argv[sys.argv.index("--test") + 1]
        config = sys.argv[sys.argv.index("--nn") + 1]
    except:
        print("Missing or invalid parameters!")
        exit(0)

    data, varnum = generate_data(train_path)
    test, _ = generate_data(test_path)

    ga = genetic_algorithm(test, crossover_average, mutate_gauss_normal, config, varnum - 1, MSE, popsize, elitism, mut_probability, mut_scale, iter_count)
    ga.train(data)