import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding3D, BatchNormalization, Flatten, Conv3D, AveragePooling3D, MaxPooling3D, GlobalMaxPooling3D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import os
import pandas as pd
import keras

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def identify_block(X,f,filters,stage,block):
    conv_name_base = 'res' +str(stage) +block +'_branch'
    bn_name_base = 'bn' + str(stage) + block +  '_branch'
    F1, F2, F3 = filters  
    X_shortcut = X 
    
    X = Conv3D(filters=F1,kernel_size=(1,1,1),strides=(1,1,1),padding='valid',name = conv_name_base+'2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name= bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    
    X = Conv3D(filters=F2, kernel_size=(f,f,f),strides=(1,1,1),padding='same',name=conv_name_base+'2b',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1,name= bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    
    X = Conv3D(filters=F3,kernel_size=(1,1,1),strides=(1,1,1),padding='valid',name=conv_name_base+'2c',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1,name=bn_name_base+'2c')(X)

    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X

def convolutional_block(X,f,filters,stage,block,s=2):
    conv_name_base = 'res' +str(stage) +block +'_branch'
    bn_name_base = 'bn' + str(stage) + block +  '_branch'
    
    F1,F2,F3 = filters
    X_shortcut = X 
    
    X = Conv3D(filters=F1,kernel_size=(1,1,1),strides=(s,s,s),padding='valid',name = conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1,name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)
 
    X = Conv3D(filters=F2,kernel_size=(f,f,f),strides=(1,1,1),padding='same',name = conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1,name= bn_name_base+'2b')(X)
    X =Activation('relu')(X)
 
    X = Conv3D(filters=F3,kernel_size=(1,1,1),strides=(1,1,1),padding='valid',name=conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name= bn_name_base+'2c')(X)
    
 
    
    X_shortcut = Conv3D(filters=F3,kernel_size=(1,1,1),strides=(s,s,s),padding='valid',name = conv_name_base+'1',
                       kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1,name = bn_name_base+'1')(X_shortcut)
    
 
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X


def ResNet50(input_shape=(33,33,33,1),classes=2):

    X_input = Input(input_shape)
    
    X = ZeroPadding3D((3,3,3))(X_input)
    
    
    X = Conv3D(filters=32,kernel_size=(3,3,3),strides=(2,2,2),name='conv1',kernel_initializer=glorot_uniform(seed=0))(X)
    X= BatchNormalization(axis=-1, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2))(X)
    
   
    X = convolutional_block(X,f=3,filters=[64,64,256],stage=2,block='a',s=1)
    X = identify_block(X, f=3,filters=[64,64,256],stage=2,block='b')
    X = identify_block(X, f=3,filters=[64,64,256],stage=2,block='c')
    
    
    X = convolutional_block(X, f=3, filters=[128,128,512], stage=3, block="a", s=2)
    X = identify_block(X, f=3, filters=[128,128,512], stage=3, block="b")
    X = identify_block(X, f=3, filters=[128,128,512], stage=3, block="c")
    X = identify_block(X, f=3, filters=[128,128,512], stage=3, block="d")

    X = convolutional_block(X, f=3, filters=[256,256,1024], stage=4, block="a", s=2)
    X = identify_block(X, f=3, filters=[256,256,1024], stage=4, block="b")
    X = identify_block(X, f=3, filters=[256,256,1024], stage=4, block="c")
    X = identify_block(X, f=3, filters=[256,256,1024], stage=4, block="d")
    X = identify_block(X, f=3, filters=[256,256,1024], stage=4, block="e")
    X = identify_block(X, f=3, filters=[256,256,1024], stage=4, block="f")
 
    
    X = convolutional_block(X, f=3, filters=[256,256,1024], stage=5, block="a", s=2)
    X = identify_block(X, f=3, filters=[256,256,1024], stage=5, block="b")
    X = identify_block(X, f=3, filters=[256,256,1024], stage=5, block="c")

    
    X = AveragePooling3D(pool_size=(2,2,2),padding='same')(X)
    
    
    
    X = Flatten()(X)
    X = Dense(classes,activation="softmax",name='fc'+str(classes),kernel_initializer=glorot_uniform(seed=0))(X)
    
    model = Model(inputs = X_input,output = X, name= 'ResNet50')
    return model


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

epoch = 40
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
tmp1 = np.zeros([33,33,33])
tmp2 = np.zeros([33,33,33])
out_name = []

for filename in files_train:
    t1 = t1+1
    npz = np.load('E:SJTU\\sjtu-m3dv-medical-3d-voxel-classification\\train_val\\'+filename)
    voxel = npz['voxel']
    seg = npz['seg']
    tmp1[0:33,0:33,0:33] = voxel[33:66,33:66,33:66]
    tmp2[0:33,0:33,0:33] = seg[33:66,33:66,33:66]
    tmp = tmp1*tmp2
    tmp_rot1 = np.rot90(tmp)
    tmp_rot2 = np.rot90(tmp,-1)
    tmp_rot3 = np.rot90(tmp,-2)
    tmp = tmp.reshape(1,tmp.shape[0],tmp.shape[1],tmp.shape[2])
    tmp_rot1 = tmp_rot1.reshape(1,tmp_rot1.shape[0],tmp_rot1.shape[1],tmp_rot1.shape[2])
    tmp_rot2 = tmp_rot2.reshape(1,tmp_rot2.shape[0],tmp_rot2.shape[1],tmp_rot2.shape[2])
    tmp_rot3 = tmp_rot3.reshape(1,tmp_rot3.shape[0],tmp_rot3.shape[1],tmp_rot3.shape[2])
    if(t1 == 1):
        x_train = tmp
        x_strength = tmp_rot1
        y_train = np.append(y_train,y_train[t1-1])
        x_strength = np.concatenate((x_strength,tmp_rot2),axis=0)
        y_train = np.append(y_train,y_train[t1-1])
        x_strength = np.concatenate((x_strength,tmp_rot3),axis=0)
        y_train = np.append(y_train,y_train[t1-1])
    else:
        x_train = np.concatenate((x_train,tmp),axis=0)
        x_strength = np.concatenate((x_strength,tmp_rot1),axis=0)
        y_train = np.append(y_train,y_train[t1-1])
        x_strength = np.concatenate((x_strength,tmp_rot2),axis=0)
        y_train = np.append(y_train,y_train[t1-1])
        x_strength = np.concatenate((x_strength,tmp_rot3),axis=0)
        y_train = np.append(y_train,y_train[t1-1])
                            
x_train = np.concatenate((x_train,x_strength),axis=0)

for filename in files_test:
    t2 = t2+1
    out_name.append(filename[:-4])
    npz = np.load('E:SJTU\\sjtu-m3dv-medical-3d-voxel-classification\\test\\'+filename)
    voxel = npz['voxel']
    seg = npz['seg']
    tmp1[0:33,0:33,0:33] = voxel[33:66,33:66,33:66]
    tmp2[0:33,0:33,0:33] = seg[33:66,33:66,33:66]
    tmp = tmp1*tmp2
    tmp = tmp.reshape(1,tmp.shape[0],tmp.shape[1],tmp.shape[2])
    if(t2 == 1):
        x_test = tmp
    else:
        x_test=np.concatenate((x_test,tmp),axis=0)

x_train = x_train.reshape(x_train.shape[0],33,33,33,1)
x_test = x_test.reshape(x_test.shape[0],33,33,33,1)
train_max = x_train.max()
train_min = x_train.min()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes)

model = ResNet50(input_shape=(33,33,33,1),classes=2)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train,y_train,epochs=50,batch_size=16,verbose=2,validation_split=0.15)


