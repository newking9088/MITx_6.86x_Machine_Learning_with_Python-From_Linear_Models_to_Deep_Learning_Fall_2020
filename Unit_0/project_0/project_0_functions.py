def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here
    A = np.random.random([n, 1])
    return A
    raise NotImplementedError

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here
    import numpy as np
    A = np.random.random([h,w])
    B = np.random.random([h,w])
    s = A + B
    return A, B, s
    raise NotImplementedError

def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    wT = np.transpose(weights)
    prod = np.matmul(wT, inputs)
    out = np.tanh(prod)
    return (out)
    # Alternatively:
    # return np.tanh(weights.T @ inputs)
    # or even:
    # return np.tanh(np.dot(inputs.T, weights))

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    if x <= y:
        return x*y
    else:
        return x/y

def get_sum_metrics(prediction, metrics=None):
    if not metrics:
        metrics = []
    for i in range(3):
        metrics.append(gen_add_i(i))

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(prediction)

    return sum_metrics

def gen_add_i(i):
    return lambda x: x + i


