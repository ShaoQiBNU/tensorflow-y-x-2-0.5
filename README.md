tensorflow训练y=x^2-0.5，构建最简单的神经网络，一个输入层，一个隐藏层（20个神经元），一个输出层
====================================================================================

# 导入库
		import tensorflow as tf
		import numpy as np

# 生成数据
## 采用np生成等差数列，（-1,1）之间，将其shape由（300，）转换为（300,1）
		
		x_data=np.linspace(-1,1,300)[:,np.newaxis]
上面也可以写成这种形式 x_data=np.linspace(-1,1,300)[:,None]		
		print(x_data.shape)

##  产生噪声，均值为0，方差为0.05的正态分布
		noise=np.random.normal(0,0.05,x_data.shape)

## y=x^2-0.5+noise
		y_data=np.square(x_data)-0.5+noise

# 占位符
		xs=tf.placeholder(tf.float32,[None,1])
		ys=tf.placeholder(tf.float32,[None,1])


# 神经网络模型
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
## 一个输入层，一个隐藏层（20个神经元），一个输出层

### 隐藏层
		h1=add_layer(xs, 1, 20, activation_function=tf.nn.relu)

### 输出层
		prediction=add_layer(h1, 20, 1, activation_function=None)

# 损失函数
## 计算输出层的预测值和真实值间的误差，对二者的差的平方求和再取平均，得到损失函数。
		loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                               reduction_indices=[1]))

其中，reduction_indices表示按哪个坐标轴求和，如图所示![](https://img-blog.csdn.net/20170617131947866?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 运用梯度下降法，以0.1的学习率最小化损失
		train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# 训练模型，训练1000次，每50次输出loss
		init=tf.global_variables_initializer()
		sess=tf.Session()
		sess.run(init)

		for i in range(1000):
    		sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    
    		if i % 50 ==0:        
        		print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
