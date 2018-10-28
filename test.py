import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tqdm import tqdm

# train, test = tf.keras.datasets.mnist.load_data()
# mnist_x, mnist_y = train
#
# mnist_ds = tf.data.Dataset.from_tensor_slices(mnist_x)
# print(mnist_ds)


from tensorflow.examples.tutorials.mnist import input_data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

def change_gaussian(m,n,t,number,x,y):
    result = np.ones((m,n,t))
    random_idx = np.random.multivariate_normal([14, 14], [[9, 0], [0, 9]], number).round().astype(int)
    for z in range(m):
        for s in random_idx:
            for i in range(y):
                for j in range(x):
                    print(j,i)
                    result[z][s[0]+j][s[1]+i] =0

    return result.reshape(m,t*n)

# tab = change_gaussian(1,28,28,5,3,3)
# data = x_train[3].reshape(1,784) * tab
#
# plt.imshow(x_train[3].reshape((28,28)))
# plt.show()
# plt.imshow(data.reshape((28,28)))
# plt.show()
#
# def change_uniform(m,n,t,number,x,y):
#     result = np.ones((m,n,t))
#     random_idx =  np.random.uniform(0., 23, size=[number,2]).round().astype(int)
#     print(random_idx)
#     for t in range(m):
#         for s in random_idx:
#             for i in range(y):
#                 for j in range(x):
#                     result[t][s[0]+j][s[1]+i] =0
#
#     return result.reshape(m,t*n)

tab = change_gaussian(1,28,28,10,3,3)
data = x_train[3].reshape(1,784) * tab

# plt.imshow(x_train[3].reshape((28,28)))
# plt.show()
plt.imshow(data.reshape((28,28)))
plt.show()




# for i in range(1000):
#     print (np.random.multivariate_normal([14,14],[[14, 0], [0, 9]],1).round().astype(int).flatten())