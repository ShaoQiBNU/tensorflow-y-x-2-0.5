# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:45:34 2018

@author: shaoqi_i
"""
############ load packages ##############
import tensorflow as tf
import numpy as np

############ generate data ##############

##### transfer x_data.shape from (300,) to (300,1) ######
##### [:,np.newaxis] also [:,None]######
x_data=np.linspace(-1,1,300)[:,np.newaxis]
#x_data=np.linspace(-1,1,300)[:,None]
print(x_data.shape)

##### add noise ######
noise=np.random.normal(0,0.05,x_data.shape)

#y=x^2-0.5+noise
y_data=np.square(x_data)-0.5+noise

##### placeholder ######
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])


############ build net model ##############
def add_layer(inputs, in_size, out_size, activation_function=None):
    
    ##### weights ######
    weights=tf.Variable(tf.random_normal([in_size, out_size]))
    
    ##### biases ######
    biases=tf.Variable(tf.zeros([1, out_size])+0.1)
    
    ##### biases ######
    Wx_plus_b=tf.matmul(inputs, weights)+biases
    
    ##### activation ######
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    
    return outputs
'''
net structure: 
    input: 1*1, 
    1 hidden layer: 1*20, 20 neurons,
    output layer: 20*1, 1 output
'''
##### hidden layer ######
h1=add_layer(xs, 1, 20, activation_function=tf.nn.relu)

##### output layer ######
prediction=add_layer(h1, 20, 1, activation_function=None)

############ loss ##############
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                               reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)


############ train ##############
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    
    if i % 50 ==0:        
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))