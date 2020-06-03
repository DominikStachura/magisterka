import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tqdm import tqdm

from load_photos import LoadPhotos

np.random.seed(1)

from tensorflow.compat.v1 import ConfigProto

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

        w1 = tf.get_variable('W1', [5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        w2 = tf.get_variable('W2', [5, 5, 32, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        w3 = tf.get_variable('W3', [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        w4 = tf.get_variable('W4', [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        w5 = tf.get_variable('W5', [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        parameters = {"W1": w1,
                      "W2": w2,
                      "W3": w3,
                      "W4": w4,
                      "W5": w5}
        return parameters

    def forward_propagation(self, X, num_classes, parameters):

        w1 = parameters['W1']
        w2 = parameters['W2']
        w3 = parameters['W3']
        w4 = parameters['W4']
        w5 = parameters['W5']

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

        # 5 convulution layer
        Z1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')
        A1 = tf.nn.relu(Z1)
        Z2 = tf.nn.conv2d(A1, w2, strides=[1, 1, 1, 1], padding='SAME')
        A2 = tf.nn.relu(Z2)
        P1 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # tutaj batch ewentualnie
        # i ewentualny dropuot 0.4

        Z3 = tf.nn.conv2d(P1, w3, strides=[1, 1, 1, 1], padding='SAME')
        A3 = tf.nn.relu(Z3)
        Z4 = tf.nn.conv2d(A3, w4, strides=[1, 1, 1, 1], padding='SAME')
        A4 = tf.nn.relu(Z4)
        P2 = tf.nn.max_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # tutaj batch ewentualnie
        # i ewentualny dropuot 0.4

        Z5 = tf.nn.conv2d(P2, w5, strides=[1, 1, 1, 1], padding='SAME')

        F1 = tf.layers.flatten(Z5)
        D1 = tf.contrib.layers.fully_connected(F1, 256)  # default activation is relu
        D2 = tf.contrib.layers.fully_connected(D1, 128)
        D3 = tf.contrib.layers.fully_connected(D2, num_classes, activation_fn=None, )

        # P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # P1 = tf.layers.flatten(P1)
        # Z3 = tf.contrib.layers.fully_connected(P1, num_classes, activation_fn=None) # dorzucicjedna warstwe gleboka
        return D3

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

        network_output = self.forward_propagation(X, self.classes, parameters)
        cost = self.compute_cost(network_output, Y)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        costs = []
        outputs = []

        class_image_dict = {i: [] for i in range(self.classes)}

        with tf.Session(config=self.config) as sess:
            sess.run(init)
            # for the first iteration
            y = np.random.random(self.classes) / 10
            for epoch in tqdm(range(num_epochs)):
                for i in range(m):
                    # for now without batches, each frame separately
                    x = X_train[i].reshape(1, self.height, self.width, self.channels)
                    _, output, cur_cost = sess.run([optimizer, network_output, cost],
                                                   feed_dict={X: x, Y: y.reshape(1, self.classes)})
                    z = sess.run(tf.argmax(tf.nn.softmax(output), 1))[0]
                    y = np.random.random(self.classes) / 10  # bliskie zeru
                    y[z] = 1
                    outputs.append(z)
                    class_image_dict[z].append(X_train[i])
                    # stworzyc macierz kwadratowa gdzie kazdej klasie uczacej odpowiadaja obrazy przypisane, zeby miec w akzdej itracji jak sie zmienialo
                    # zrobic dicta zeby miec dla kazdego neuronu przypisany obrazek i potem splot po kazdej
                    # epoce tych obrazkow, zbey zobaczyc jakie ksztalty sie pojawiaja
                    # {'klasa': [obrazki],...}
                # print(y)
                costs.append(cur_cost)
            saver = tf.train.Saver(tf.global_variables())
            print('Fit Done')
            saver.save(sess, 'trained_model')
            with open('class_image_dict.pickle', 'wb') as f:
                pickle.dump(class_image_dict, f)

            # _, counts = np.unique(outputs, return_counts=True)
            # plt.bar(np.arange(self.classes), counts)
            # plt.xticks(np.arange(self.classes))
            # plt.xlabel('Counts')
            # plt.ylabel('Neuron with the highest value')
            # plt.show()
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
    load = LoadPhotos(r'datasets', end_with='.png')
    load_test = LoadPhotos(r'datasets', begin_with='test', end_with='.png')
    X_train = load.load(64, 64)  # moze byc mniejsze np 64 lub 32
    # X_test = load_test.load(128, 128)
    shape = X_train.shape
    print(f'{shape[0]} photos loaded of shape {shape[1:]}')
    np.random.shuffle(X_train)
    model = NetworkModel(picture_size=shape[1:], classes=3)
    model.fit(X_train, learning_rate=0.1,
              num_epochs=25)  # learning na poczatku moze byc duzy, nwe 0.5, i zmniejszac dynamcznie
    # zmieniac dynamicznie np dzielic na 2 co 5 epok
    # model.predict(X_test[1])
