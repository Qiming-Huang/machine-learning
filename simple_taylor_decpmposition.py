import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten
import matplotlib.pyplot as plt

class Simple_Tayplor_Decomposition():
    def __init__(self, model_path, data_path):
        self.data = np.load(data_path)
        self.model = load_model(model_path)
        self.x_train, self.y_train = self.data['x_train'] / 255, self.data['y_train']
        self.X = tf.Variable(self.get_sample_img(), dtype=tf.float32)

    def get_sample_img(self):
        sample_img = []
        for i in range(10):
            img = self.x_train[self.y_train == i]
            sample_img.append(img[np.random.choice(img.shape[0])])

        return sample_img

    def show(self, save=False):
        with tf.GradientTape() as tape:
            Y = self.model(self.X)

        result_taylor = self.X * tape.gradient(Y, self.X)

        fig, axs = plt.subplots(2, 5)		# axs
        num = 0
        fig.suptitle('simple taylor decomposition')
        for row in axs:
            for col in row:
                col.imshow(result_taylor[num], cmap='bwr')
                col.set_title('number' + str(num))
                num += 1
        if save:
            plt.savefig('simple_taylor_decomposition_result.jpg')
        plt.show()

    def train(self, epochs=3, batch_size=32, save=False):
        x = Input(shape=(28, 28, 1))

        net = Flatten()(x)
        net = Dense(512, activation='relu', use_bias=False)(net)
        net = Dense(512, activation='relu', use_bias=False)(net)

        output = Dense(10)(net)
        model = Model(inputs=x, outputs=output)

        x_train = self.x_train[:, :, :, np.newaxis]
        y_train = tf.one_hot(self.y_train, dtype=tf.float32, depth=10)


        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        if save:
            model.save('model.h5')


if __name__ == '__main__':
    model_path = 'model.h5'
    data_path = 'mnist.npz'
    std = Simple_Tayplor_Decomposition(model_path, data_path)
    std.show(save=False)
    std.train(save=False)
