from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
y = boston.target/100
print(y)
print(X)

print(X.shape)
print(y.shape)

from Model import *

learning_rate = 1e-2
model = models()
model.add(Dense(input_size=None, hidden_layer_num=128, activation=activations.sigmoid))
model.add(Dense(input_size=None, hidden_layer_num=64, activation=activations.sigmoid))
model.add(Dense(input_size=None, hidden_layer_num=32, activation=activations.sigmoid))
model.add(Dense(input_size=None, hidden_layer_num=16, activation=activations.sigmoid))
model.add(Dense(input_size=None, hidden_layer_num=8, activation=activations.sigmoid))
model.add(Dense(input_size=None, hidden_layer_num=1, activation=activations.sigmoid))

model.fit(train_data=X, train_labels=y, shuffle=False, epochs=100, batch_size=100,
          learning_rate=learning_rate, measure=measure.mse)

model.plot()
