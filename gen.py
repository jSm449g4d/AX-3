#functionalAPI_test
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
import argparse
import os
from datetime import datetime


#from AR9
def mkdiring(input):
    arr=input.split("/");input=""
    for inp in arr:
        if not os.path.exists(input+inp+"/"):os.mkdir(input+inp)
        input+=inp+"/"
    return input.rstrip("/")

def tf2img(tfs,dir="./",name="",epoch=0,ext=".png"):
    mkdiring(dir)
    if type(tfs)!=np.ndarray:tfs=tfs.numpy()
    tfs=(tfs*256).astype(np.uint8)
    for i in range(tfs.shape[0]):
        cv2.imwrite(os.path.join(dir,name+"_epoch-num_"+str(epoch)+"-"+str(i)+ext),tfs[i])

def tf_ini():#About GPU resources
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
    if len(physical_devices)==0:print("GPU_failed!")
    return len(physical_devices)

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train' ,help="train_data",default="./lfw")
parser.add_argument('-o', '--outdir' ,help="outdir",default="./output")
parser.add_argument('-b', '--batch' ,help="batch",default=32,type=int)
parser.add_argument('-p', '--predbatch' ,help="batch_size_of_prediction",default=8,type=int)
parser.add_argument('-e', '--epoch' ,help="epochs",default=50,type=int)
parser.add_argument('--smodel' ,help="saved_model",default="./output/gen.h5")
args = parser.parse_args()

if __name__ == '__main__':
    tf_ini()
    try:
        gans=keras.models.load_model(args.smodel)
        outs=gans.predict(np.random.rand(4,8).astype(np.float32))
        tf2img(outs,os.path.join(args.outdir,"dep"),
               epoch=int(datetime.now().timestamp()))
    except:print("error")
    
    
    