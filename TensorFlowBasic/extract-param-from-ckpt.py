import numpy as np
import tensorflow as tf

ckpt_path=tf.train.latest_checkpoint('my_net')

reader = tf.train.NewCheckpointReader(ckpt_path)

#独处ckpt_path中的变量信息
variables = reader.get_variable_to_shape_map() 

#临时保存用的字典
dic={}

for ele in variables:

	#多个层次的分隔符,因此我们在定义Layer的时候不能起名带有_的层次
	layer_name=ele.split('_')[0]

	#根据名称找到tensor数值
	temp=np.array(reader.get_tensor(ele))

	#添加key : value (键值:layer  value:权重)
	if not dic.setdefault(layer_name):

		#添加到字典中
		dic[layer_name]=[temp,]

	else:

		# 添加到字典中
		dic[layer_name].append(temp)

#字典转换numpy
param=np.array(dic)

#保存到npy文件
np.save('mnist.npy',param)

	
