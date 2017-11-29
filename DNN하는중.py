#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:20:48 2017

@author: moonsooyoung
"""
import os
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pylab as plt
import PIL.Image as pilimg

#MNIST 숫자데이터를 불러온다.
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#DeepNeuralNetwork에 쓸 각 layer에 사용할 노드의 갯수를 사용자가 원하는 만큼 정해준다.
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#n_classes = 데이터의 클래스 수 입니다. MNIST에서는 0,1,2,3,4,5,6,7,8,9 총 10개 이다.
n_classes = 10

#batch_size를 원하는 만큼 정해준다.
batch_size = 100

#input과 output의 placeholder을 만들어줍니다. [None, 784]는 input data의 사이즈이다.
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

#신경망 구조에 들어갈 변수를 정의해줍니다. 훈련 전에는 랜덤한 값을 임의로 넣어 초기값을 잡아준다.
#y=XW+b 임을 고려하여 벡터크기를 지정해준다.
w1 = tf.Variable(tf.random_normal([784, n_nodes_hl1]))
b1 = tf.Variable(tf.random_normal([n_nodes_hl1]))

w2 = tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]))
b2 = tf.Variable(tf.random_normal([n_nodes_hl2]))

w3 = tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]))
b3 = tf.Variable(tf.random_normal([n_nodes_hl3]))

w4 = tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]))
b4 = tf.Variable(tf.random_normal([n_classes]))

#신경망 구조
l1 = tf.add(tf.matmul(x,w1), b1)
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1,w2), b2)
l2 = tf.nn.relu(l2)

l3 = tf.add(tf.matmul(l2,w3), b3)
l3 = tf.nn.relu(l3)

output = tf.matmul(l3,w4) + b4


#변수들을 저장해줄 파일경로를 지정해준다
save_path = 'pyhelp/'
model_name = 'cnnmodel'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path_full = os.path.join(save_path, model_name)

#학습 단계 알고리즘
def train_neural_network(x):
    prediction = output
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                sess.run(l1, feed_dict={x: epoch_x})
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        #아까 지정한 파일경로로 변수들을 저장해준다. 지정할 변수를 리스트 형태로 써준다.
        saver = tf.train.Saver([w1, w2, w3, w4, b1, b2, b3, b4])
        saver.save(sess,save_path_full)
        
#학습 시작
train_neural_network(x)


#테스트 단계
#테스트할 파일을 불러온다
k = pilimg.open('/Users/moonsooyoung/Desktop/pyhelp/msoo.png' )
plt.imshow(k)
imgarray=np.array(k)    #컬러채널이 1개가 되도록 이미 전처리를 한 상태라서 그냥 (28,28)로 나온다.
kkk = imgarray/255    #k1 벡터 안의 숫자들을 0과 1 사이로 normalize시키기 벡터 내의 가장 큰 값으로 k1을 나눠준다.
sydata=kkk.reshape(1,784)    #훈련할 때와 같은 input data 형태로 맞춰준다 (28,28)->(1,784)
x_data = tf.cast(sydata, 'float')

#변수 불러오기
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver([w1, w2, w3, w4, b1, b2, b3, b4])    #전에 저장해둔 변수를 불러온다.
new_saver = tf.train.import_meta_graph('/Users/moonsooyoung/Desktop/pyhelp/pyhelp/cnnmodel.meta')    
#학습 단계에서 자동으로 생긴 meta file에 변수들의 값이 저장되어 있다. saver.restore()로 그 값들을 불러온다
saver.restore(sess,save_path_full)

#처음에 썼던 신경망 알고리즘을 그대로 써주고, input data넣는 x자리에 테스트 할 데이터(x_data)를 바꿔 써준다.
l1 = tf.add(tf.matmul(x_data,w1), b1)
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1,w2), b2)
l2 = tf.nn.relu(l2)

l3 = tf.add(tf.matmul(l2,w3), b3)
l3 = tf.nn.relu(l3)

output = tf.matmul(l3,w4) + b4
print('============================================TEST 결과============================================')
print(sess.run (output))
print(sess.run(tf.argmax(output, 1)))
