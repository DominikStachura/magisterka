import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from load_photos import LoadPhotos
from tqdm import tqdm

np.random.seed(1)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# config = tf.ConfigProto(
#         device_count = {'CPU': 0}
#     )
config = ConfigProto()
config.gpu_options.allow_growth = True


class NetworkModel:
    def __init__(self, picture_size, classes):
        self.height, self.width, self.channels = picture_size
        self.classes = classes
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True

    def create_placeholders(self):

        X = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels), name='input')
        Y = tf.placeholder(tf.float32, shape=(None, self.classes))

        return X, Y

    def initialize_weights(self):
        tf.set_random_seed(1)

        w1 = tf.get_variable('W1', [4, 4, 3, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        w2 = tf.get_variable('W2', [2, 2, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        parameters = {"W1": w1,
                      "W2": w2}
        return parameters

    def forward_propagation(self, X, num_classes, parameters):

        w1 = parameters['W1']
        w2 = parameters['W2']

        # # Conv 1
        # Z1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')
        # #     Z1 = tf.nn.dropout(Z1, keep_prob=0.5)
        # # RELU
        # A1 = tf.nn.relu(Z1)
        # # MAXPOOL
        # P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # # CONV2
        # Z2 = tf.nn.conv2d(P1, w2, strides=[1, 1, 1, 1], padding='SAME')
        # #     Z2 = tf.nn.dropout(Z2, keep_prob=0.5)
        # # RELU
        # A2 = tf.nn.relu(Z2)
        # # MAXPOOL
        # P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # # FLATTEN
        # P2 = tf.layers.flatten(P2)
        # # FC
        # Z3 = tf.contrib.layers.fully_connected(P2, num_classes, activation_fn=None)

        #shallow architecture
        Z1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')
        A1 = tf.nn.relu(Z1)
        P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        P1 = tf.layers.flatten(P1)
        Z3 = tf.contrib.layers.fully_connected(P1, num_classes, activation_fn=None) # dorzucicjedna warstwe gleboka
        return Z3
    # na razie bez dropouta

    def compute_cost(self, Z3, Y):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
        return cost

    def fit(self, X_train, learning_rate=0.01, num_epochs=10):
        ops.reset_default_graph()
        tf.set_random_seed(0)
        # number of train examples
        m = len(X_train)
        # Create Placeholders of the correct shape
        X, Y = self.create_placeholders()

        # Initialize parameters
        parameters = self.initialize_weights()

        Z3 = self.forward_propagation(X, self.classes, parameters)
        cost = self.compute_cost(Z3, Y)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        costs = []
        outputs = []
        with tf.Session(config=self.config) as sess:
            sess.run(init)
            # for the first iteration
            y = np.random.random(self.classes)
            for epoch in tqdm(range(num_epochs)):
                for i in range(m):
                    # for now without batches, each frame separately
                    x = X_train[i].reshape(1, self.height, self.width, self.channels)
                    _, Z, cur_cost = sess.run([optimizer, Z3, cost], feed_dict={X: x, Y: y.reshape(1, self.classes)})
                    z = sess.run(tf.argmax(tf.nn.softmax(Z), 1))[0]
                    # Z = sess.run(tf.nn.softmax(Z3), feed_dict={X: x})
                    # Z = sess.run(Z3, feed_dict={X: x})
                    y = np.random.random(self.classes)/10 #bliskie zeru
                    y[z] = 1
                    outputs.append(z)
                    # print(Z)
                    # print(z)
                    # stworzyc macierz kwadratowa gdzie kazdej klasie uczacej odpowiadaja obrazy przypisane, zeby miec w akzdej itracji jak sie zmienialo
                    # zrobic dicta zeby miec dla kazdego neuronu przypisany obrazek i potem splot po kazdej
                    # epoce tych obrazkow, zbey zobaczyc jakie ksztalty sie pojawiaja
                    #{'klasa': [obrazki],...}
                # print(y)
                costs.append(cur_cost)
            saver = tf.train.Saver(tf.global_variables())
            print('Fit Done')
            saver.save(sess, 'trained_model')
            _, counts = np.unique(outputs, return_counts=True)
            plt.bar(np.arange(self.classes), counts)
            plt.xticks(np.arange(self.classes))
            plt.xlabel('Counts')
            plt.ylabel('Neuron with the highest value')
            plt.show()
            # try:
            #     plt.savefig('6_classes_10_outputs.png')
            # except:
            #     pass

    def predict(self, test_frame):
        tf.reset_default_graph()
        with tf.Session(config=self.config) as sess:
            # load the computation graph (the fully connected + placeholder)
            loader = tf.train.import_meta_graph('trained_model.meta')
            sess.run(tf.global_variables_initializer())

            X = tf.get_default_graph().get_tensor_by_name('input:0')
            activation = tf.get_default_graph().get_tensor_by_name('fully_connected/BiasAdd:0')
            # now load the weights
            loader.restore(sess, 'trained_model')
            actual = sess.run(activation, {X: test_frame.reshape(1, self.height, self.width, self.channels)})
            print(f'Prediction:\n{sess.run(tf.argmax(tf.nn.softmax(actual), axis=1))}\n')


if __name__ == "__main__":
    load = LoadPhotos(r'./mis/',  end_with='.png')
    load_test = LoadPhotos(r'./test/', begin_with='test', end_with='.png')
    X_train = load.load(128, 128) # moze byc mniejsze np 64 lub 32
    X_test = load_test.load(128, 128)
    shape = X_train.shape
    print(f'{shape[0]} photos loaded of shape {shape[1:]}')
    np.random.shuffle(X_train)
    model = NetworkModel(picture_size=shape[1:], classes=10)
    model.fit(X_train, learning_rate=0.0001, num_epochs=30) # learning na poczatku moze byc duzy, nwe 0.5, i zmniejszac dynamcznie
                                                            #zmieniac dynamicznie np dzielic na 2 co 5 epok
    # model.predict(X_test[1])
