# numpy-NN
Neural Network implementation from scratch using numpy

## Usage

#### numpy-nn
```
model = Network()
model.add(DenseLayer(6))
model.add(DenseLayer(8))
model.add(DenseLayer(3))

model.train(X_train=X, y_train=y, epochs=200)
```
> EPOCH: 0, ACCURACY: 0.3333333333333333, LOSS: 1.8507288360616592
>
> EPOCH: 20, ACCURACY: 0.64, LOSS: 0.8984484293696664
>
> EPOCH: 40, ACCURACY: 0.5666666666666667, LOSS: 0.8055846210908157
>
> EPOCH: 60, ACCURACY: 0.5933333333333334, LOSS: 0.7544998303196496
>
> EPOCH: 80, ACCURACY: 0.6466666666666666, LOSS: 0.7034754660535022
>
> EPOCH: 100, ACCURACY: 0.8666666666666667, LOSS: 0.6522870909240465
>
> EPOCH: 120, ACCURACY: 0.9466666666666667, LOSS: 0.6051327850621049
>
> EPOCH: 140, ACCURACY: 0.96, LOSS: 0.5624822108029988
>
> EPOCH: 160, ACCURACY: 0.96, LOSS: 0.5237726663927962
>
> EPOCH: 180, ACCURACY: 0.9533333333333334, LOSS: 0.4887972804949555
>

#### keras equivalent
```
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

ohy = tf.keras.utils.to_categorical(y, num_classes=3)

model2 = Sequential()
model2.add(Dense(6, activation='relu'))
model2.add(Dense(10, activation='relu'))
model2.add(Dense(8, activation='relu'))
model2.add(Dense(3, activation='softmax'))

model2.compile(SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(x=X, y=ohy, epochs=30)
```

## Metrics
![Accuracy](https://github.com/j0sephsasson/numpy-nn/blob/main/accuracy.png?raw=true)
![Loss](https://github.com/j0sephsasson/numpy-nn/blob/main/loss.png?raw=true)

## Notes
 - Currently: 
    - relu activation in hidden layers
    - softmax activation in the output layer
    - cross-entropy loss
    - multi-class tasks *ONLY*

## Next Steps
1. Documentation
2. Binary classification tasks
3. Regression tasks
4. Multiple loss functions / optimizations