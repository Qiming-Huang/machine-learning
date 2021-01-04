import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt



model = load_model('model.h5')
data = np.load('mnist.npz')

x_train, y_train = data['x_train'], data['y_train']
x_train = x_train.reshape((60000, 28, 28, 1)) / 255

X_prototype = tf.random.normal(shape=(10, 28, 28, 1))
Y_prototype = tf.one_hot(tf.cast(tf.linspace(0., 9., 10), tf.int32), depth=10)
X_prototype = tf.Variable(X_prototype)
Y_prototype = tf.Variable(Y_prototype)
lmda = 0.1

# fig, axs = plt.subplots(2, 5)
# fig.suptitle('Random img')
# num = 0
# for i in axs:
#     for j in i:
#         if num <= 9:
#             j.imshow(X_prototype[num])
#             j.set_title('number' + str(num))
#         num += 1
# plt.savefig('b.jpg')
# plt.show()

img_means = []
for i in range(10):
    img_means.append(np.mean(x_train[y_train == i], axis=0))

for epoch in range(1000):
    logits_prototype = model(X_prototype)
    loss = lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_prototype, labels=Y_prototype)) \
                   + lmda * tf.nn.l2_loss(X_prototype - img_means)
    opt = tf.keras.optimizers.SGD(lr=0.1)
    opt.minimize(loss, var_list=[X_prototype])
    print("this is %d epoch, the loss is %f" % (epoch, loss()))

fig, axs = plt.subplots(2, 5)
fig.suptitle('Activation Maximization')
num = 0
for i in axs:
    for j in i:
        if num <= 9:
            j.imshow(X_prototype[num])
            j.set_title('number' + str(num))
        num += 1
plt.savefig('a.jpg')
plt.show()
