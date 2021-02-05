from math import exp

def logistic(x):
    return 1 / (1 + exp(-x))

def dx_logistic(x):
    return exp(-x) / ((1 + exp(-x)) ** 2)

def hiperbolic_tan(x):
    return (1 - exp(-2 * x)) / (1 + exp(-2 * x))
