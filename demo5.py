from Model import *

# # 从这里开始编写网络
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('C:/Users/JDUSER/Documents/我的坚果云/代码/tftyd/data/', one_hot=True)
train_data = mnist.train.next_batch(60000)
test_data = mnist.train.next_batch(10000)
train_x = train_data[0]
train_y = train_data[1]

test_x = test_data[0]
test_y = test_data[1]

learning_rate = 1e-4
model = models()
model.add(Dense(input_size=None, hidden_layer_num=128, activation=activations.relu))
model.add(Dense(input_size=None, hidden_layer_num=128, activation=activations.relu))
model.add(Dense(input_size=None, hidden_layer_num=10, activation=activations.softmax))
for i in range(30):
    print('step',i)
    model.fit(train_data=train_x, train_labels=train_y, epochs=1, batch_size=20,
              learning_rate=learning_rate, measure=measure.cross_entropy, shuffle=True)
    model.verify(test_data=test_x, test_labels=test_y, batch_size=20)
model.plot()
