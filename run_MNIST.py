import mnist
import experiment_runner
import FLattack

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

root_data,root_label,clients_data,clients_label,x_test,y_test = mnist.Load_MNIST(q = 0.1, nclient = 100, root_dataset_size = 100, root_dataset_bias = 0.1)
print(x_test.shape)
# for i in range(0,100):
#     print(clients_data[i].shape)
server1 = experiment_runner.run_no_attacks(root_data,root_label,clients_data,clients_label, 100, x_test, y_test)
server2 = experiment_runner.run_only_server(root_data,root_label, x_test, y_test)

FLattack_label = FLattack.LabelFlippingAttack(clients_label=clients_label, nclients = 100, malicious_ratio = 0.2)
server3 = experiment_runner.run_no_attacks(root_data,root_label,clients_data,FLattack_label, 100, x_test, y_test)


server2.evaluate(root_data,root_label, 'train_on_server train')
server2.evaluate(x_test,y_test, 'train_on_server test')
server1.evaluate(x_test,y_test, 'FLTrust in test data')
server3.evaluate(x_test,y_test, 'FLTrust in test data with FL attack')











# import sysp
# import os
# CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
# config_path = CURRENT_DIR.rsplit('/', 2)[0]  # 上三级目录
# sys.path.append(config_path)


# import experiments.mnist.experiment_runner as experiment_runner

# experiment_runner.run_no_attacks('expr_no_attacks', seed=1, cpr='all', rounds=100, mu=1.5, sigma=3.45, alpha=0.1,
#                                  t_mean_beta=0.1)

# experiment_runner.run_no_attacks('expr_no_attacks', seed=1, cpr='all', rounds=100, mu=1.5, sigma=3.45, alpha=0.1,
#                                  t_mean_beta=0.1)

# experiment_runner.run_all('expr_to_zero_10_precent', seed=1, cpr='all', rounds=100, mu=1.5, sigma=3.45, real_alpha=0.1,
#                           num_samples_per_attacker=
#                           1_000_000, attack_type='delta_to_zero', alpha=0.1, t_mean_beta=0.1)

# experiment_runner.run_all('expr_to_y_flip_10_precent', seed=1, cpr='all', rounds=100, mu=1.5, sigma=3.45,
#                           real_alpha=0.1,
#                           num_samples_per_attacker=1_000_000, attack_type='y_flip', alpha=0.1, t_mean_beta=0.1)

# experiment_runner.run_all('expr_to_zero_single', seed=1, cpr='all', rounds=100, mu=1.5, sigma=3.45, real_alpha=0.1,
#                           num_samples_per_attacker=10_000_000, attack_type='delta_to_zero', alpha=1, t_mean_beta=0.1,
#                           real_alpha_as_f=True)

# experiment_runner.run_all('expr_y_flip_single', seed=1, cpr='all', rounds=100, mu=1.5, sigma=3.45,
#                           real_alpha=0.1,
#                           num_samples_per_attacker=10_000_000, attack_type='y_flip', alpha=1, t_mean_beta=0.1,
#                           real_alpha_as_f=True)
