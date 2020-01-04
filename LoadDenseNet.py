import numpy as np
import tensorflow as tf
import os
import pandas as pd
import keras
from keras.utils.data_utils import get_file
from keras.models import Model, save_model, load_model
import keras.backend as K
 


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

path_test = ('E:\\SJTU\\sjtu-m3dv-medical-3d-voxel-classification\\test')
files_test = os.listdir(path_test)
files_test.sort(key=lambda x: int(x[9:-4]))
csv_file = pd.read_csv('E:SJTU\\sjtu-m3dv-medical-3d-voxel-classification\\train_val.csv')
mask = csv_file['lable']
t2 = 0
tmp1 = np.zeros([32,32,32])
tmp2 = np.zeros([32,32,32])
out_name = []
        

for filename in files_test:
    t2 = t2+1
    out_name.append(filename[:-4])
    npz = np.load('E:SJTU\\sjtu-m3dv-medical-3d-voxel-classification\\test\\'+filename)
    voxel = npz['voxel']
    seg = npz['seg']
    tmp1[0:32,0:32,0:32] = voxel[33:65,33:65,33:65]
    tmp2[0:32,0:32,0:32] = seg[33:65,33:65,33:65]
    tmp = tmp1*tmp2
    tmp = tmp.reshape(1,tmp.shape[0],tmp.shape[1],tmp.shape[2])
    if(t2 == 1):
        x_test = tmp
    else:
        x_test=np.row_stack((x_test,tmp))


x_test = x_test.reshape(x_test.shape[0],32,32,32,1)
x_test = x_test.astype('float32') / 255

model = load_model("DenseNet.h5")
result = model.predict(x_test)
result = np.reshape(result,(117,))
out_name = np.array(out_name)
data_frame = pd.DataFrame({'Id':out_name,'Predicted':result})
data_frame.to_csv('Submission.csv')
