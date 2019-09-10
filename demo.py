from Model import *

# # 从这里开始编写网络
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100
mnist = input_data.read_data_sets('C:/Users/JDUSER/Documents/我的坚果云/代码/tftyd/data/', one_hot=True)
data = mnist.train.next_batch(batch_size)
x = data[0]
y = data[1]

model = models()
model.add(Dense(input_size=None, hidden_layer_num=32, activation=activations.relu))
model.add(Dense(input_size=None, hidden_layer_num=10, activation=activations.softmax))

model.batch_input = x
model.batch_label = y
model.summary()
# '先使用一组数据做测试'
model.learning_rate = 0.0003

for i in range(10000):
    data = mnist.train.next_batch(batch_size)
    x = data[0]
    y = data[1]
    model.batch_input = x
    model.batch_label = y
    model.forward()
    model.backward()
model.plot()
