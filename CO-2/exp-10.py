import tensorflow as tf
from tensorflow.keras import layers, regularizers

# Baseline Overfitting Model vs Regularized Model
X = np.random.randn(500, 20)
y = np.random.randint(0, 2, size=500)

model_reg = tf.keras.Sequential([
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

model_reg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_reg.fit(X, y, epochs=100, validation_split=0.3, callbacks=[callback], verbose=0)
print(f"Stopped at epoch: {len(history.history['loss'])}")
