import numpy as np



modelPath = "vgg16.npy"

#读取model参数
wDict = np.load(modelPath, encoding = "bytes").item()

#那么就是表征层的名称
for name in wDict:

    print(name)

    # for p in wDict[name]:
    #
    #     #显示对应Layer层的权重或者其他参数数值
    #     print (p)