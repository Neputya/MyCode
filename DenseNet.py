import numpy as np
import tensorflow as tf
import os
import pandas as pd
import keras
from keras.layers import Input, Dropout, LeakyReLU, Dense, BatchNormalization, concatenate
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D
from keras.models import Model, load_model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.models import Model, save_model, load_model
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import keras.backend as K

def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
    
    # Bottleneck layers
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv3D(bn_size*nb_filter, (1,1,1), strides=(1,1,1), padding='same')(x)
    
    # Composite function
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv3D(nb_filter, (3,3,3), strides=(1,1,1), padding='same')(x)
    
    if drop_rate: x = Dropout(drop_rate)(x)
    
    return x
 
def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=-1)
        
    return x
    
def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):
    
    nb_filter = int(x.shape.as_list()[-1]*compression)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv3D(nb_filter, (1,1,1), strides=(1,1,1), padding='same')(x)
    if is_max != 0: x = MaxPooling3D(pool_size=(2,2,2), strides=2)(x)
    else: x = AveragePooling3D(pool_size=(2,2,2), strides=2)(x)
    
    return x

def scheduler(epoch):
    if (epoch %10 ==0 and epoch > 0):
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.01)
        print("lr changed to {}".format(lr * 0.01))
    return K.get_value(model.optimizer.lr)


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

epoch = 30
batch_size = 8
num_classes = 2
path_train = ('E:\\SJTU\\sjtu-m3dv-medical-3d-voxel-classification\\train_val')
path_test = ('E:\\SJTU\\sjtu-m3dv-medical-3d-voxel-classification\\test')
files_train = os.listdir(path_train)
files_train.sort(key=lambda x: int(x[9:-4]))
files_test = os.listdir(path_test)
files_test.sort(key=lambda x: int(x[9:-4]))
csv_file = pd.read_csv('E:SJTU\\sjtu-m3dv-medical-3d-voxel-classification\\train_val.csv')
mask = csv_file['lable']
y_train = np.array(mask)
y_train.reshape(1,-1)
t1 = 0
t2 = 0
tmp1 = np.zeros([32,32,32])
tmp2 = np.zeros([32,32,32])
out_name = []

for filename in files_train:
    t1 = t1+1
    npz = np.load('E:SJTU\\sjtu-m3dv-medical-3d-voxel-classification\\train_val\\'+filename)
    voxel = npz['voxel']
    seg = npz['seg']
    tmp1[0:32,0:32,0:32] = voxel[33:65,33:65,33:65]
    tmp2[0:32,0:32,0:32] = seg[33:65,33:65,33:65]
    tmp = tmp1*tmp2
    tmp = tmp.reshape(1,tmp.shape[0],tmp.shape[1],tmp.shape[2])

    if(t1 == 1):
        x_train = tmp
    else:
        x_train = np.row_stack((x_train,tmp))
        

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

x_train = x_train.reshape(x_train.shape[0],32,32,32,1)
x_test = x_test.reshape(x_test.shape[0],32,32,32,1)
train_max = x_train.max()
train_min = x_train.min()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32') / 255
#y_train = keras.utils.to_categorical(y_train, num_classes)

growth_rate = 12
 
inpt = Input(shape=(32,32,32,1))
 
x = Conv3D(growth_rate*2, (3,3,3), strides=1, padding='same')(inpt)
x = BatchNormalization(axis=-1)(x)
x = LeakyReLU(alpha=0.1)(x)

x = DenseBlock(x, 4, growth_rate, drop_rate=0.1)
 
x = TransitionLayer(x)
 
x = DenseBlock(x, 4, growth_rate, drop_rate=0.1)
 
x = TransitionLayer(x)
 
x = DenseBlock(x, 4, growth_rate, drop_rate=0.1)
 
x = BatchNormalization(axis=-1)(x)
x = GlobalAveragePooling3D()(x)
 
x = Dense(1, activation='sigmoid')(x)

reduce_lr = LearningRateScheduler(scheduler) 
model = Model(inpt, x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,batch_size=8,verbose=2,validation_split=0.1,callbacks=[reduce_lr])

save_model(model, 'DenseNet1.h5')
result = model.predict(x_test)
result = np.reshape(result,(117,))
out_name = np.array(out_name)
data_frame = pd.DataFrame({'Id':out_name,'Predicted':result})
data_frame.to_csv('E:SJTU\\sjtu-m3dv-medical-3d-voxel-classification\\Submission1.csv')
