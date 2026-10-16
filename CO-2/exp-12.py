import tensorflow as tf
from tensorflow.keras import layers, models
import time

def evaluate_cnn(pool_type='max', use_bn=True):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3,3), input_shape=(32,32,3)))
    if use_bn: model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    if pool_type == 'max':
        model.add(layers.MaxPooling2D((2,2)))
    else:
        model.add(layers.AveragePooling2D((2,2)))
        
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    history = model.fit(X_train[:1000], y_train[:1000], epochs=2, verbose=0)
    dur = time.time() - start
    return model.count_params(), dur

params, dur = evaluate_cnn(pool_type='max', use_bn=True)
print(f"Max Pool + BN Params: {params}, Execution Time: {dur:.2f}s")
