import numpy as np
import tensorflow as tf

ckpt_path=tf.train.latest_checkpoint('my_net')
reader = tf.train.NewCheckpointReader(ckpt_path)
variables = reader.get_variable_to_shape_map() 

dic={}
for ele in variables:
	layer_name=ele.split('_')[0]
	temp=np.array(reader.get_tensor(ele))
	if not dic.setdefault(layer_name):
		dic[layer_name]=[temp,]
	else:
		dic[layer_name].append(temp)
param=np.array(dic)
np.save('capsnet_para.npy',param)

	
