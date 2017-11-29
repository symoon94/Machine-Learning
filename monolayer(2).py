
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:19:42 2017

@author: moonsooyoung
"""
import numpy as np
import matplotlib.pylab as plt
import os
import tensorflow as tf
import cv2

x=tf.placeholder('float', [None, 784])
y=tf.placeholder('float', [None, 10])

W = tf.Variable(tf.zeros([784,10]), name='W')
b = tf.Variable(tf.zeros([10]), name = 'b')

#변수들을 저장해줄 파일경로를 지정해준다
save_path = 'pyhelp/'
model_name = 'modelsoo1'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
save_path_full = os.path.join(save_path, model_name)


#테스트단계
img_path = '/Users/moonsooyoung/Desktop/Python_4sy/msoo.png'
k = cv2.resize(cv2.imread(img_path ), (28, 28)).astype(np.float32)

plt.imshow(k)
kk=k[:,:,2]
plt.imshow(kk)
kkk=kk.reshape(1,784)
k4 = kkk/255
data=tf.cast(k4, 'float')

sess= tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess,save_path_full)
TEST= tf.nn.softmax(tf.matmul(data,W)+b)
print('*****************************************************')
print(sess.run(TEST))
print(sess.run(tf.argmax(TEST,1)))
