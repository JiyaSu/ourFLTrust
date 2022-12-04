from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import client 
import server 

def CNN_model_factory():
  model = models.Sequential()
  model.add(layers.Conv2D(30, (3, 3), activation='relu', input_shape=(28, 28, 1)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(50, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(100, activation='relu'))
  model.add(layers.Dense(10,activation='softmax'))
  return model
  
def run_no_attacks(root_data, root_label, clients_data, clients_label, nclient, x_test, y_test):
  beta  = 5e-2
  alpha = 5e-2/beta
  model_factory = CNN_model_factory
  clients = []
  for i in range(0,nclient):
    clients.append(client.Client(model_factory, clients_data[i], clients_label[i], learning_rate=beta, R=1, batch_size=32))
  root_client = client.Client(model_factory, root_data, root_label, learning_rate=beta, R=1, batch_size=32)
  server1 = server.Server(model_factory, select_client=nclient, iteration=250, alpha=alpha, beta=beta)
  server1.train(clients,root_client,'no_attacks',x_test, y_test)
  return server1

def run_only_server(root_data, root_label, x_test, y_test):
  beta  = 5e-2
  alpha = 5e-2/beta
  model_factory = CNN_model_factory
  server1 = server.Server(model_factory, select_client=100, iteration=250, alpha=alpha, beta=beta)
  root_client = client.Client(model_factory, root_data, root_label, learning_rate=beta, R=1, batch_size=32)
  server1.train_self(root_client,'train_on_server', x_test, y_test)
  return server1













# import itertools
# import json
# import pathlib
# import random
# from dataclasses import dataclass, field
# from functools import partial

# import numpy as np
# import tensorflow as tf

# from shared.aggregators import mean, median, trimmed_mean
# from shared.truncate import find_U
# import experiments.mnist.mnist as mnist
# from experiments.mnist.client import Client
# from experiments.mnist.server import Server


# def fs_setup(experiment_name, seed, config):
#   root_dir = pathlib.Path(f'experiments') / experiment_name
#   config_path = root_dir / 'config.json'

#   # get model config
#   if config_path.is_file():
#     with config_path.open() as f:
#       stored_config = json.load(f)

#       if json.dumps(stored_config, sort_keys=True) != json.dumps(config, sort_keys=True):
#         with (root_dir / 'config_other.json').open(mode='w') as f_other:
#           json.dump(config, f_other, sort_keys=True, indent=2)
#         raise Exception('stored config should equal run_experiment\'s parameters')
#   else:
#     root_dir.mkdir(parents=True, exist_ok=True)
#     with config_path.open(mode='w') as f:
#       json.dump(config, f, sort_keys=True, indent=2)

#   experiment_dir = root_dir / f'seed_{seed}'
#   experiment_dir.mkdir(parents=True, exist_ok=True)

#   return experiment_dir


# def run_experiment(experiment_name, seed, model_factory, server_config,
#                    partition_config, num_of_rounds, threat_model):
#   server = Server(model_factory, **server_config)

#   experiment_dir = fs_setup(experiment_name, seed, {
#     # 'model': server.model.get_config(),
#     'partition_config': partition_config
#   })

#   expr_basename = f'{server_config["weight_delta_aggregator"].__name__}' \
#                   f'{server_config["clients_importance_preprocess"].prefix}' \
#                   f'_cpr_{server_config["clients_per_round"]}' \
#                   f'{(threat_model.prefix if threat_model is not None else "")}'
#   expr_file = experiment_dir / f'{expr_basename}.npz'
#   if expr_file.is_file():
#     prev_results = np.load(expr_file, allow_pickle=True)
#     server_weights = prev_results['server_weights'].tolist()
#     server.model.set_weights(server_weights)
#     history = prev_results['history'].tolist()
#     start_round = len(history)
#     if start_round >= num_of_rounds:
#       print(f'skipping {expr_basename} (seed={seed}) '
#             f'start_round({start_round}), num_of_rounds({num_of_rounds})')
#       return
#   else:
#     history = []
#     start_round = 0

#   np.random.seed(seed)
#   tf.random.set_seed(seed)
#   random.seed(seed)

#   (partitioned_x_train, partitioned_y_train), (test_x, test_y) = mnist.load(partition_config)

#   clients = [
#     Client(i, data, model_factory)
#     for i, data in enumerate(zip(partitioned_x_train, partitioned_y_train))
#   ]
#   if threat_model is not None:
#     attackers = np.random.choice(
#         clients,
#         int(len(clients) * threat_model.real_alpha) if threat_model.real_alpha is not None else int(threat_model.f),
#         replace=False)
#     for client in attackers:
#       client.as_attacker(threat_model)

#   server.train(clients, test_x, test_y, start_round, num_of_rounds, expr_basename, history,
#                lambda history, server_weights: np.savez(expr_file, history=history, server_weights=server_weights))


# def CNN_model_factory():
#   model = models.Sequential()
#   model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#   return tf.keras.models.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation='softmax')
#   ])


# def passthrough_preprocess(_): return _


# passthrough_preprocess.prefix = '_w'


# def ignore_weights_preprocess(_): return None


# ignore_weights_preprocess.prefix = ''


# def truncate_preprocess(num_of_samples_per_client, alpha):
#   U = find_U(np.array(num_of_samples_per_client), alpha=alpha)
#   return [min(U, num_of_samples) for num_of_samples in num_of_samples_per_client] if U else num_of_samples_per_client


# def truncate_preprocess_with_alpha(alpha):
#   preprocess = partial(truncate_preprocess, alpha=alpha)
#   preprocess.prefix = f'_t_{int(preprocess.keywords["alpha"] * 100)}'
#   return preprocess


# @dataclass(frozen=True)
# class Threat_model:
#   type: str
#   num_samples_per_attacker: int
#   real_alpha: int = None
#   f: int = None
#   prefix: str = field(init=False)

#   def __post_init__(self):
#     object.__setattr__(self, 'prefix',
#                        f'_b_{self.type}_'
#                        f'{int(self.real_alpha * 100) if self.real_alpha is not None else "f" + str(self.f)}_'
#                        f'{self.num_samples_per_attacker}')


# def run_no_attacks(experiment, seed, cpr, rounds, mu, sigma, alpha, t_mean_beta):
#   t_mean = partial(trimmed_mean, beta=t_mean_beta)
#   t_mean.__name__ = f't_mean_{int(t_mean_beta * 100)}'

#   weight_delta_aggregators = [t_mean, median, mean]
#   preprocessors = [truncate_preprocess_with_alpha(alpha=alpha), ignore_weights_preprocess, passthrough_preprocess]

#   for (wda, preprocessor) in itertools.product(weight_delta_aggregators, preprocessors):
#     run_experiment(experiment,
#                    seed=seed,
#                    model_factory=mlp_model_factory,
#                    server_config={
#                      'clients_importance_preprocess': preprocessor,
#                      'weight_delta_aggregator': wda,
#                      'clients_per_round': cpr,
#                    },
#                    partition_config={'#clients': 100, 'mu': mu, 'sigma': sigma},
#                    num_of_rounds=rounds,
#                    threat_model=None
#                    )


# def run_all(experiment, seed, cpr, rounds, mu, sigma, real_alpha, num_samples_per_attacker, attack_type='y_flip',
#             alpha=0.1, t_mean_beta=0.1, real_alpha_as_f=False):
#   t_mean = partial(trimmed_mean, beta=t_mean_beta)
#   t_mean.__name__ = f't_mean_{int(t_mean_beta * 100)}'

#   weight_delta_aggregators = [mean, t_mean, median]
#   preprocessors = [passthrough_preprocess, truncate_preprocess_with_alpha(alpha=alpha), ignore_weights_preprocess]

#   threat_models = [None] if attack_type is None else [
#     Threat_model(type=attack_type, num_samples_per_attacker=num_samples_per_attacker,
#                  f=real_alpha) if real_alpha_as_f else Threat_model(type=attack_type,
#                                                                     num_samples_per_attacker=num_samples_per_attacker,
#                                                                     real_alpha=real_alpha),
#   ]

#   for (threat_model, wda, preprocessor) in itertools.product(threat_models, weight_delta_aggregators, preprocessors):
#     run_experiment(experiment,
#                    seed=seed,
#                    model_factory=mlp_model_factory,
#                    server_config={
#                      'clients_importance_preprocess': preprocessor,
#                      'weight_delta_aggregator': wda,
#                      'clients_per_round': cpr,
#                    },
#                    partition_config={'#clients': 100, 'mu': mu, 'sigma': sigma},
#                    num_of_rounds=rounds,
#                    threat_model=threat_model
#                    )
