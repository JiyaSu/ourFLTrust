import random

import numpy as np
import tensorflow as tf


def MNIST_DataLabel(data,label):
  labels = np.zeros([10,10])
  for i in range(0,10):
      labels[i][i] = 1
      
  labeled_data = [[],[],[],[],[],[],[],[],[],[]]
  for i in range(0,len(label)):
    label_int = int(label[i])
    labeled_data[label_int].append(data[i])
    
  for i in range(0,10):
    random.shuffle(labeled_data[i])
    
  return labeled_data,labels

def Root_Dataset(labeled_data,labels,root_dataset_size,root_dataset_bias):
  root_data = []
  root_label = []
  idx = np.zeros([10],np.int)
  for i in range(0,root_dataset_size):
    rand = random.random()
    if rand < root_dataset_bias:
      label = 0
    else:
      label = random.randint(0,8)+1
    root_data.append(labeled_data[label][idx[label]])
    idx[label] += 1
    root_label.append(labels[label])
  return root_data,root_label,idx

def Choose_client(l, q, nclass, nclient):
  group_size = nclient/nclass
  rand = random.random()
  if rand < q:
    label = l
  else:
    label = random.randint(0,8)
    if label >= l:
      label += 1
  group_idx = random.randint(0,group_size-1)
  client = label*group_size+group_idx
  # print(label,group_idx)
  return int(client)



def Load_MNIST(q, nclient = 100, root_dataset_size=100, root_dataset_bias=0.1):
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  labeled_data,labels = MNIST_DataLabel(x_train, y_train)
  root_data,root_label,idx = Root_Dataset(labeled_data,labels,root_dataset_size,root_dataset_bias)

  nclass = 10
  clients_data = []
  clients_label = []
  for i in range(0,nclient):
    clients_data.append([]);
    clients_label.append([]);
    
  for i in range(0,nclass):
    for j in range(idx[i],len(labeled_data[i])):
      client_id = Choose_client(i,q,nclass,nclient)
      clients_data[client_id].append(labeled_data[i][j])
      clients_label[client_id].append(labels[i])

  root_data = np.array(root_data)
  root_data = root_data.reshape((root_data.shape[0], 28, 28, 1))
  root_label = np.array(root_label)
  # root_label = root_label.reshape((root_label.shape[0], 10, 1))
  for i in range(0,nclient):
    clients_data[i] = np.array(clients_data[i])
    clients_data[i] = clients_data[i].reshape((clients_data[i].shape[0], 28, 28, 1))
    clients_label[i] = np.array(clients_label[i])
    # clients_label[i] = clients_label[i].reshape((clients_label[i].shape[0],10,1))

  y_test = tf.keras.utils.to_categorical(y_test)
  # y_test = y_test.reshape((y_test.shape[0],10,1))
  x_test = np.array(x_test)
  x_test = x_test.reshape((x_test.shape[0],28,28,1))
  
  return root_data,root_label,clients_data,clients_label,x_test,y_test
  
  
  
def Load_FashionMNIST(q, nclient = 100, root_dataset_size=100, root_dataset_bias=0.1):
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  labeled_data,labels = MNIST_DataLabel(x_train, y_train)
  root_data,root_label,idx = Root_Dataset(labeled_data,labels,root_dataset_size,root_dataset_bias)

  nclass = 10
  clients_data = []
  clients_label = []
  for i in range(0,nclient):
    clients_data.append([]);
    clients_label.append([]);
    
  for i in range(0,nclass):
    for j in range(idx[i],len(labeled_data[i])):
      client_id = Choose_client(i,q,nclass,nclient)
      clients_data[client_id].append(labeled_data[i][j])
      clients_label[client_id].append(labels[i])

  root_data = np.array(root_data)
  root_data = root_data.reshape((root_data.shape[0], 28, 28, 1))
  root_label = np.array(root_label)
  # root_label = root_label.reshape((root_label.shape[0], 10, 1))
  for i in range(0,nclient):
    clients_data[i] = np.array(clients_data[i])
    clients_data[i] = clients_data[i].reshape((clients_data[i].shape[0], 28, 28, 1))
    clients_label[i] = np.array(clients_label[i])
    # clients_label[i] = clients_label[i].reshape((clients_label[i].shape[0],10,1))

  y_test = tf.keras.utils.to_categorical(y_test)
  # y_test = y_test.reshape((y_test.shape[0],10,1))
  x_test = np.array(x_test)
  x_test = x_test.reshape((x_test.shape[0],28,28,1))
  
  return root_data,root_label,clients_data,clients_label,x_test,y_test


