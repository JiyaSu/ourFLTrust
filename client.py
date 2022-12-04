import tensorflow as tf
import numpy as np

class Client:
  def __init__(self, model_factory, data, label, learning_rate, R, batch_size):
    self.attacker = False
    self.threat_model = None
    self._x = data
    self._y = label
    self._model = model_factory()
    self.learning_rate = learning_rate
    self.epochs = R
    self.batch_size = batch_size
    self.steps_per_epoch = 1

  def train(self, server_weights):
    self._model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
      loss = tf.keras.losses.CategoricalCrossentropy(),
      metrics=['accuracy']
    )
    self._model.set_weights(server_weights)
    self._model.fit(x = self._x, y = self._y, verbose=0,
                    epochs = self.epochs, batch_size = self.batch_size, 
                    steps_per_epoch=self.steps_per_epoch,
                    )
    new_weights = self._model.get_weights()
    delta_weights = [new_w - old_w for new_w, old_w in zip(new_weights, server_weights)]

    return delta_weights


# class Client():
#   def __init__(self, idx, data, model_factory):
#     self.idx = idx

#     self.attacker = False
#     self.threat_model = None

#     self.num_of_samples = len(data[0])

#     self._x, self._y = data[0], data[1]

#     self._model = model_factory()

#   def as_attacker(self, threat_model):
#     self.attacker = True
#     self.threat_model = threat_model

#     if self.threat_model.type == 'y_flip':
#       self._y = 9 - self._y

#     self.num_of_samples = self.threat_model.num_samples_per_attacker

#   def train(self, server_weights):
#     if self.attacker and self.threat_model is not None and self.threat_model.type == 'delta_to_zero':
#       return [-_ for _ in server_weights]

#     self._model.compile(
#       optimizer=tf.keras.optimizers.SGD(learning_rate=5e-2),
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     )

#     self._model.set_weights(server_weights)

#     self._model.fit(self._x, self._y, verbose=0,
#                     # go over 10% of data like in Yin's paper
#                     epochs=1, batch_size=max((len(self._x) // 10), 1), steps_per_epoch=1,
#                     # epochs=3, batch_size=50,
#                     #                         callbacks=[tf.keras.callbacks.EarlyStopping(
#                     #                             monitor='loss', patience=1, restore_best_weights=True)]
#                     )

#     new_weights = self._model.get_weights()

#     delta_weights = [new_w - old_w for new_w, old_w in zip(new_weights, server_weights)]

#     if self.attacker and self.threat_model is not None and self.threat_model.type == 'sign_flip':
#       return [-t for t in delta_weights]
#     else:
#       return delta_weights
