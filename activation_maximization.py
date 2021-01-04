import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

class ActivationMaximization():
    """
    the implementation of Activation Maximization, it used a pre-trained model to find a
    maximum model response for a quantity of interest.
    used a random vector to generate the maximum response of activation.
    """
    def __init__(self, model_path, data_path):
        """
        initialize variables
        :param model_path: str, the file path of the pre-trained moedl (.h5)
        :param data_path: str, the file path of the training data (mnist.npz)
        """
        self.model_path = model_path
        self.data_path = data_path
        self.X_prototype = tf.Variable(tf.random.normal(shape=(10, 28, 28, 1)))
        self.Y_prototype = tf.Variable(tf.one_hot(tf.cast(tf.linspace(0., 9., 10), tf.int32), depth=10))
        self.lmda = 0.1
        self.x_train, self.y_train, self.model = self.preprocessing()
        self.img_means = self.imges_means()

    def preprocessing(self):
        """
        get some data
        :return: x_train, y_train, model
        """
        data = np.load(self.data_path)
        x_train, y_train = data['x_train'] / 255, data['y_train']
        model = load_model(self.model_path)

        return x_train, y_train, model

    def loss(self, logits_prototype):
        """
        get loss function
        :param logits_prototype: the reference result given by model(self.X_prototype)
        :return: loss function
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_prototype, labels=self.Y_prototype)) \
               + self.lmda * tf.nn.l2_loss(self.X_prototype - self.img_means)

        return loss

    def imges_means(self):
        """
        calculate the mean of pre iamge
        :return: img_means np.array
        """
        img_means = []
        for i in range(10):
            img_means.append(np.mean(self.x_train[self.y_train == i], axis=0))
        img_means = np.array(img_means)
        return img_means[:, :, :, np.newaxis]

    def get_optimizer(self, lr=0.1):
        """
        get optimizer
        :param lr: learning rate, default 0.1
        :return: tf.keras.optimizers
        """
        return tf.keras.optimizers.SGD(lr)

    def train_with_keras(self, epochs):
        """
        update the imput pixels using optimizer.minimize
        note, in this case, the loss function have to be determined by lambda function, otherwise
        it will appear uncallable error, I don't know why.
        :param epochs: training epochs
        :return: none
        """
        optimizer = self.get_optimizer()
        for epoch in range(epochs):
            logits_prototype = self.model(self.X_prototype)
            am_loss = lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_prototype, labels=self.Y_prototype)) \
                              + self.lmda * tf.nn.l2_loss(self.X_prototype - self.img_means)
            optimizer.minimize(am_loss, var_list=[self.X_prototype])
            print("this is %d epoch, the loss is %f" % (epoch, am_loss()))

    def train_with_gradient_tape(self, epochs):
        """
        update the imput pixels using tf.GradientTape()
        :param epochs: training epochs
        :return: none
        """
        optimizer = self.get_optimizer()
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                logits_prototype = self.model(self.X_prototype)
                am_loss = self.loss(logits_prototype)
            gradients = tape.gradient(am_loss, [self.X_prototype])  # X_prototype has to be included by two Brackets!!!!
            optimizer.apply_gradients(zip(gradients, [self.X_prototype]))  # X_prototype has to be included by two Brackets!!!!
            print("this is %d epoch, the loss is %f" % (epoch, am_loss))


if __name__ == '__main__':
    model_path = 'model.h5'
    data_path = 'mnist.npz'
    am = ActivationMaximization(model_path, data_path)
    am.train_with_gradient_tape(100)
    # am.train_with_keras(100)
