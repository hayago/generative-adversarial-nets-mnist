import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class GAN:
    """
    Generative Adversarial Networks
    """
    IMAGE_SIZE = 784  # 28 x 28 = 784

    def __init__(self):
        # ---------- Generator's variables ----------
        self.Z = tf.placeholder(tf.float32, shape=[None, 100])

        self.G_W1 = tf.Variable(xavier_initial_value([100, 128]))
        self.G_b1 = tf.Variable(tf.zeros([128]))

        self.G_W2 = tf.Variable(xavier_initial_value([128, GAN.IMAGE_SIZE]))
        self.G_b2 = tf.Variable(tf.zeros([GAN.IMAGE_SIZE]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        # ---------- Discriminator's variables ----------
        self.X = tf.placeholder(tf.float32, shape=[None, GAN.IMAGE_SIZE])

        self.D_W1 = tf.Variable(xavier_initial_value([GAN.IMAGE_SIZE, 128]))
        self.D_b1 = tf.Variable(tf.zeros([128]))

        self.D_W2 = tf.Variable(xavier_initial_value([128, 1]))
        self.D_b2 = tf.Variable(tf.zeros([1]))

        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

    def train(self):
        G_sample = self.__generator(self.Z)

        D_real = self.__descriminator(self.X)
        D_fake = self.__descriminator(G_sample)

        # Loss function
        D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        G_loss = tf.reduce_mean(tf.log(1. - D_fake))

        D_train_step = tf.train.AdamOptimizer(1e-5).minimize(D_loss, var_list=self.theta_D)
        G_train_step = tf.train.AdamOptimizer(1e-5).minimize(G_loss, var_list=self.theta_G)

        # Load data sets
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        if not os.path.exists('output/'):
            os.makedirs('output/')

        batch_size = 100
        Z_dim = 100
        plot_count = 1000
        for i in range(1000000):
            batch, _ = mnist.train.next_batch(batch_size)

            sess.run([D_train_step, G_train_step], feed_dict={
                self.Z: sample_random_uniform([batch_size, Z_dim]),
                self.X: batch
            })

            if i % plot_count == 0:
                D_loss_current, G_loss_current = sess.run([D_loss, G_loss], feed_dict={
                    self.Z: sample_random_uniform([batch_size, Z_dim]),
                    self.X: batch
                })

                print('Iteration: {}'.format(i))
                print('D loss: {:.10}'.format(D_loss_current))
                print('G loss: {:.10}'.format(G_loss_current))
                print()

                samples = sess.run(G_sample, feed_dict={
                    self.Z: sample_random_uniform([16, Z_dim])
                })

                # plot
                fig = plot(samples)
                plt.savefig('output/{}.png'.format(i // plot_count).zfill(4), bbox_inches='tight')
                plt.close(fig)

    def __generator(self, z):
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.sigmoid(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        return G_h2

    def __descriminator(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_h2 = tf.nn.sigmoid(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        return D_h2


def xavier_initial_value(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(size, stddev=xavier_stddev)


def sample_random_uniform(size):
    return np.random.uniform(-1., 1., size=size)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(samples[i].reshape(28, 28), cmap='Greys_r')

    return fig


if __name__ == '__main__':
    gan = GAN()
    gan.train()
