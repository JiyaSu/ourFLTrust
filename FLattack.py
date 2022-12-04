import copy
import numpy as np


def LabelFlippingAttack(clients_label, nclients, malicious_ratio):
    m = int(nclients * malicious_ratio)
    attack_label = copy.deepcopy(clients_label)
    
    clientsID = np.zeros([nclients],np.int)
    for i in range(0,nclients):
        clientsID[i] = i
    attack_clients = np.random.choice(clientsID,m,replace=False)
    # print(attack_clients)

    for i in range(0,len(attack_clients)):
        attack_clientID = attack_clients[i]
        for j in range(0,len(attack_label[attack_clientID])):
            nclass = len(attack_label[attack_clientID][j])
#            print(nclass)
            for l in range(0,nclass):
                if attack_label[attack_clientID][j][l] == 1:
                    attack_label[attack_clientID][j][l] = 0
                    attack_label[attack_clientID][j][nclass-l-1] = 0
                    break
    
    return attack_label



