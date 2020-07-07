## 基于numpy实现的全连接深度学习工具
实现以下功能:

- **激活函数**:relu,sigmoid,tanh,softmax
- **损失函数**:平均绝对误差(MAE),交叉熵(cross_entropy)
- **前向传播**:前向传播(forward),反向传播(backward 使用梯度下降优化器) 
- **权重矩阵**:权重矩阵形状推导和模型总结(summary)  
- **随机函数**:随机函数(shuffle)      
- **预测函数**:输出模型的预测结果(predict)              
- **验证损失**:使用模型预测并计算损失(verify)    
- **绘制曲线**:绘制损失变化曲线(plot)

在mnist数据集上的准确率为0.847
使用封装的fit训练函数训练
``` python
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
``` 

手动进行前向传播和反向传播训练
``` python
model = models()
model.add(Dense(input_size=None, hidden_layer_num=32, activation=activations.relu))
model.add(Dense(input_size=None, hidden_layer_num=10, activation=activations.softmax))

model.batch_input = x
model.batch_label = y
model.summary()
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
``` 

> demo提供了训练mnist数据的完整示例