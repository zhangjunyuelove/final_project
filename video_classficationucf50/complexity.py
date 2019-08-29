from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
model_dir = "E:/cifar-10/cifar10-CNN-master/cifar10_train_2fl"
checkpoint_path = os.path.join(model_dir, "model.ckpt-150000")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
total_parameters = 0
for key in var_to_shape_map:#list the keys of the model
    # print(key)
    # print(reader.get_tensor(key))
    shape = np.shape(reader.get_tensor(key))  #get the shape of the tensor in the model
    shape = list(shape)
    # print(shape)
    # print(len(shape))
    variable_parameters = 1
    for dim in shape:
        # print(dim)
        variable_parameters *= dim
    # print(variable_parameters)
    total_parameters += variable_parameters

print(total_parameters)
