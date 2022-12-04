import mnist
import experiment_runner
import FLattack

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

root_data,root_label,clients_data,clients_label,x_test,y_test = mnist.Load_FashionMNIST(q = 0.5, nclient = 100, root_dataset_size = 100, root_dataset_bias = 0.1)
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


