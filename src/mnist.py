import numpy as np
from torchvision.datasets import mnist
from itertools import product
mnist_train = mnist.MNIST(".", download=True, train=True)
mnist_test = mnist.MNIST(".", download=True, train=False)

img_size = 28
x_train = np.asarray(mnist_train.train_data, dtype='float32')
x_train = x_train.reshape([len(x_train), -1])
y_train = mnist_train.train_labels.numpy()


x_val = np.asarray(mnist_test.test_data, dtype='float32')
x_val = x_val.reshape([len(x_val), -1])
y_val = mnist_test.test_labels.numpy()

_grid = (np.arange(img_size) + 0.5 ) / img_size
grid_y, grid_x = map(np.asarray, zip(*product(_grid, _grid)))

def make_cloud(row, size=100, noise=1e-2):
    """
    Samples a cloud of points
    :param row: image pixels, treated as probabilities
    :param size: how many points to generate
    :param noise: additional gitter added to points
    :returns: point coordinates : points_x, points_y
    :rtype: np.ndarray, float [size,]
    """
    if row.dtype != 'float32':
        row = row.astype('float32')
    if row.shape != 1:
        row = row.flatten()
    
    assert row.shape == grid_x.shape
    
    point_indices = np.random.choice(len(row), p=row / row.sum(), size=size)
    
    point_x = grid_x[point_indices] + np.random.normal(0, noise, size)
    point_y = grid_y[point_indices] + np.random.normal(0, noise, size)
    
    return point_x, point_y

def make_clouds(rows, size=100, noise=1e-2):
    """
    Samples several clouds in parallel with make_cloud
    :returns: point clouds of shape [n_rows, size, 2]
    """
    return np.apply_along_axis(lambda x: make_cloud(x, size, noise), 1, rows).transpose(0, 2, 1)
    