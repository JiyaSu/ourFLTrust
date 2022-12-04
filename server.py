import numpy as np
import tensorflow as tf
import copy

class Server:
  def __init__(self, model_factory, select_client, iteration, alpha, beta):
    self._model = model_factory()
    self.nselect_client = select_client
    self.global_itera = iteration
    self.alpha = alpha
    self._self_model = model_factory()
    self._model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=beta),
      loss = tf.keras.losses.CategoricalCrossentropy(),
      metrics=['accuracy']
    )

  def train_self(self, root_client, expr_basename, x_test, y_test):
    for r in range(0, self.global_itera):
      server_weights = self._model.get_weights()
      root_delta = root_client.train(server_weights)
      new_server_weights = [self.alpha*delta_w + old_w for delta_w, old_w in zip(root_delta, server_weights)]
      self._model.set_weights(new_server_weights)
      print(f'{expr_basename} round={r + 1}/{self.global_itera}',
              end='')
      print('')
      loss, acc = self._model.evaluate(x_test, y_test, batch_size=32)
      print(f'{expr_basename} loss: {loss} - accuracy: {acc:.2%}')


  def evaluate(self, x_test, y_test, expr_basename):
    loss, acc = self._model.evaluate(x_test, y_test, verbose=0)
    print(f'{expr_basename} loss: {loss} - accuracy: {acc:.2%}')




    
  def train(self, clients, root_client, expr_basename, x_test, y_test):
    for r in range(0, self.global_itera):
      server_weights = self._model.get_weights()
      new_server_weights = copy.deepcopy(server_weights)
      # print(server_weights)
      selected_clients = clients if self.nselect_client == len(clients) \
        else np.random.choice(clients, self.nselect_client, replace=False)
      deltas = []
      for i, client in enumerate(selected_clients):
        print(f'{expr_basename} round={r + 1}/{self.global_itera}, client {i + 1}/{self.nselect_client}',
              end='')
        deltas.append(client.train(server_weights))
        if i != len(selected_clients) - 1:
          print('\r', end='')
        else:
          print('')

      root_delta = root_client.train(server_weights)

      tmp_server_weights = copy.deepcopy(root_delta)
      tmp_server_weights = np.concatenate([x.ravel() for x in tmp_server_weights])
      # print(tmp_server_weights.shape)
      total_TS = 0
      TSnorm = []
      for d in deltas:
        tmp_weight = copy.deepcopy(d)
        tmp_weight = np.concatenate([x.ravel() for x in tmp_weight])
        TS = np.dot(tmp_weight,tmp_server_weights)/(np.linalg.norm(tmp_weight)*np.linalg.norm(tmp_server_weights))
        if TS < 0:
          TS = 0
        total_TS += TS

        norm = np.linalg.norm(tmp_server_weights)/np.linalg.norm(tmp_weight)
        TSnorm.append(TS*norm)
      
      # print(TSnorm[0])
      # print(deltas[0])
      delta_weight = [TSnorm[0]*x for x in deltas[0]]
      for i in range(1,len(deltas)):
        for j in range(0,len(delta_weight)):
          delta_weight[j] += TSnorm[i]*deltas[i][j]
      
      for j in range(0,len(delta_weight)):
        delta_weight[j] /= total_TS

      # print(delta_weight)
      
      for j in range(0,len(delta_weight)):
        new_server_weights[j] += self.alpha*delta_weight[j]
      self._model.set_weights(new_server_weights)
      
      print(f'{expr_basename} round={r + 1}/{self.global_itera}',
              end='')
      print('')
      loss, acc = self._model.evaluate(x_test, y_test, batch_size=32)
      print(f'{expr_basename} loss: {loss} - accuracy: {acc:.2%}')

    



# class Server:
#   def __init__(self, model_factory, clients_importance_preprocess, weight_delta_aggregator, clients_per_round):
#     self._clients_importance_preprocess = clients_importance_preprocess
#     self._weight_delta_aggregator = weight_delta_aggregator
#     self._clients_per_round = clients_per_round if clients_per_round == 'all' else int(clients_per_round)

#     self.model = model_factory()

#     self.model.compile(
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#       metrics=['accuracy']
#     )

#   def train(self, clients, test_x, test_y, start_round, num_of_rounds, expr_basename, history, progress_callback):
#     client2importance = self._clients_importance_preprocess([c.num_of_samples for c in clients])

#     server_weights = self.model.get_weights()

#     for r in range(start_round, num_of_rounds):
#       selected_clients = clients if self._clients_per_round == 'all' \
#         else np.random.choice(clients, self._clients_per_round, replace=False)

#       deltas = []
#       for i, client in enumerate(selected_clients):
#         print(f'{expr_basename} round={r + 1}/{num_of_rounds}, client {i + 1}/{self._clients_per_round}',
#               end='')

#         deltas.append(client.train(server_weights))

#         if i != len(selected_clients) - 1:
#           print('\r', end='')
#         else:
#           print('')

#       if client2importance is not None:
#         importance_weights = [client2importance[c.idx] for c in selected_clients]
#       else:
#         importance_weights = None

#       # todo change code below (to be nicer?):
#       # aggregated_deltas = [self._weight_delta_aggregator(_, importance_weights) for _ in zip(*deltas)]
#       # server_weights = [w + d for w, d in zip(server_weights, aggregated_deltas)]
#       server_weights = [w + self._weight_delta_aggregator([d[i] for d in deltas], importance_weights)
#                         for i, w in enumerate(server_weights)]

#       self.model.set_weights(server_weights)
#       loss, acc = self.model.evaluate(test_x, test_y, verbose=0)
#       print(f'{expr_basename} loss: {loss} - accuracy: {acc:.2%}')
#       history.append((loss, acc))
#       if (r + 1) % 10 == 0:
#         progress_callback(history, server_weights)
