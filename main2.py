#functionalAPI_test
#main.py -> functionalAPI_ize ->mode seeking
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
from tqdm import tqdm
import argparse
import os
import random


#from AR9
def mkdiring(input):
    arr=input.split("/");input=""
    for inp in arr:
        if not os.path.exists(input+inp+"/"):os.mkdir(input+inp)
        input+=inp+"/"
    return input.rstrip("/")
def ffzk(input_dir):#Relative directory for all existing files
    imgname_array=[];input_dir=input_dir.strip("\"\'")
    for fd_path, _, sb_file in os.walk(input_dir):
        for fil in sb_file:imgname_array.append(fd_path.replace('\\','/') + '/' + fil)
    if os.path.isfile(input_dir):imgname_array.append(input_dir.replace('\\','/'))
    return imgname_array

def img2np(dir=[],img_len=0):
    img=[]
    for x in dir:
        try:img.append(cv2.imread(x))
        except:continue
        if img_len!=0:img[-1]=cv2.resize(img[-1],(img_len,img_len))
        elif img[-1].shape!=img[0].shape:img.pop(-1);continue#Leave only the same shape
        img[-1] = img[-1].astype(np.float32)/ 256
    return np.stack(img, axis=0)

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

class c3c():
    def __init__(self,dim=4):#plz define used layers below...
        self.dim=dim
        return
    def __call__(self,mod):#plz add layers below...
        with tf.name_scope("c3c"):
            mod_1=mod
            mod=Conv2D(self.dim,1,padding="same")(mod)
            mod_1=Conv2D(self.dim,3,padding="same")(mod_1)
            mod_1=Conv2D(self.dim,3,padding="same")(mod_1)
            mod_1=Conv2D(self.dim,3,padding="same")(mod_1)
            mod=keras.layers.add([mod,mod_1])
            mod=Dropout(0.05)(mod)
            mod=Activation("relu")(mod)
        return mod
    
def GEN(input_dim=8):
    mod=mod_inp = Input(shape=(input_dim,))
    mod=Reshape((1,1,8))(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(12)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(16)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(24)(mod)
    mod=c3c(32)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(32)(mod)
    mod=c3c(48)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(48)(mod)
    mod=c3c(54)(mod)
    mod=c3c(64)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(64)(mod)
    mod=c3c(72)(mod)
    mod=c3c(84)(mod)
    mod=c3c(94)(mod)
    mod=Conv2D(3,1,padding="same",activation="sigmoid")(mod)
    return keras.models.Model(inputs=mod_inp, outputs=mod)

def DIS(input_shape=(64,64,3,)):
    mod=mod_inp = Input(shape=input_shape)
    mod=c3c(16)(mod)
    mod=c3c(18)(mod)
    mod=c3c(24)(mod)
    mod=Conv2D(24,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=c3c(28)(mod)
    mod=c3c(32)(mod)
    mod=c3c(36)(mod)
    mod=Conv2D(36,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=c3c(42)(mod)
    mod=c3c(46)(mod)
    mod=c3c(48)(mod)
    mod=Conv2D(48,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=c3c(54)(mod)
    mod=c3c(64)(mod)
    mod=Conv2D(64,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=c3c(84)(mod)
    mod=c3c(84)(mod)
    mod=Conv2D(84,4,2,padding="same",activation="relu")(mod)
    mod=c3c(84)(mod)
    mod=LayerNormalization()(mod)
    mod=Dropout(0.05)(mod)
    mod=Flatten()(mod)
    mod=Dense(84,activation="relu")(mod)
    mod=Dense(64,activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=Dropout(0.05)(mod)
    mod=Dense(32,activation="relu")(mod)
    mod=Dense(1,activation="sigmoid")(mod)
    return keras.models.Model(inputs=mod_inp, outputs=mod)
    
class gan():
    def __init__(self,trials=[],dim=8):
        self.dim=dim
        self.gen=GEN(input_dim=self.dim)
        self.dis=DIS()
    def pred(self,batch=4):
        return self.gen(np.random.rand(batch,self.dim).astype(np.float32))
    def train(self,data=[],epoch=1000,batch=16,predbatch=8):
        
        optimizer = keras.optimizers.SGD(0.003)
        self.gen.compile(optimizer = optimizer,
                          loss=keras.losses.binary_crossentropy)
        self.dis.compile(optimizer = optimizer,
                          loss=keras.losses.binary_crossentropy)
        self.gen.summary()
        self.dis.summary()
        
        try:self.gen.load_weights(os.path.join(args.outdir,"disw.h5"))
        except:print("\nCannot_use_savedata...")
        try:self.dis.load_weights(os.path.join(args.outdir,"genw.h5"))
        except:print("\nCannot_use_savedata...")
                
        ones=np.ones(batch).astype(np.float32)        
        zeros=np.zeros(batch).astype(np.float32)
        labels=np.random.rand(data.shape[0],self.dim).astype(np.float32)
        
        
        for i in range(epoch):
            with tqdm(total=data.shape[0]) as pbar:
                while pbar.n+batch<pbar.total:
                    datum=data[pbar.n:pbar.n+batch];label=labels[pbar.n:pbar.n+batch];
                
                    with tf.GradientTape() as tape:
                        dis=self.dis(self.gen(label))
                        dis=keras.losses.binary_crossentropy(zeros,dis)
                        dis=tf.reduce_mean(dis) 
                        grad=tape.gradient(dis,self.dis.trainable_variables)
                        grad,_ = tf.clip_by_global_norm(grad, 15)
                        optimizer.apply_gradients(zip(grad,self.dis.trainable_variables))
                        del tape
                    
                    with tf.GradientTape() as tape:
                        dis=self.dis(datum)
                        dis=keras.losses.binary_crossentropy(ones,dis)
                        dis=tf.reduce_mean(dis) 
                        grad=tape.gradient(dis,self.dis.trainable_variables)
                        grad,_ = tf.clip_by_global_norm(grad, 15)
                        optimizer.apply_gradients(zip(grad,self.dis.trainable_variables))
                        del tape
                    
                    with tf.GradientTape() as tape:
                        gen=self.gen(label)
                        dis_gen=self.dis(self.gen(label))
                        dis_gen=keras.losses.binary_crossentropy(ones,dis_gen)
                        dis_gen=tf.reduce_mean(dis_gen) 
                        ####mode seeking (tentative implement)###
                        roll_shift=random.randint(1,batch-1)
                        Lms=tf.reduce_mean(tf.abs(label-tf.roll(label,roll_shift,axis=0)),[1])
                        Lms/=tf.reduce_mean(tf.abs(gen-tf.roll(gen,roll_shift,axis=0)),[1,2,3])+np.full(batch,1e-5)
                        dis_gen+=tf.reduce_mean(Lms) 
                        ####mode seeking (tentative implement)###
                        grad=tape.gradient(dis_gen,self.gen.trainable_variables) 
                        grad,_ = tf.clip_by_global_norm(grad, 15)
                        optimizer.apply_gradients(zip(grad,self.gen.trainable_variables))
                        del tape
                    
                    pbar.update(batch)
                    
            print("\nke(fake,tru):",
                  tf.reduce_mean(keras.losses.binary_crossentropy (zeros,self.dis(self.gen(labels[:batch])))).numpy(),
                  tf.reduce_mean(keras.losses.binary_crossentropy (ones,self.dis(data[:batch])))).numpy()
            tf2img(self.pred(predbatch),os.path.join(args.outdir,"1"),epoch=i,ext=".png")
                    
            self.dis.save_weights(os.path.join(args.outdir,"disw.h5"))
            self.gen.save_weights(os.path.join(args.outdir,"genw.h5"))
            self.gen.save(os.path.join(args.outdir,"gen.h5"))
        

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train' ,help="train_data",default="./lfw")
parser.add_argument('-o', '--outdir' ,help="outdir",default="./output")
parser.add_argument('-b', '--batch' ,help="batch",default=32,type=int)
parser.add_argument('-p', '--predbatch' ,help="batch_size_of_prediction",default=8,type=int)
parser.add_argument('-e', '--epoch' ,help="epochs",default=50,type=int)
args = parser.parse_args()

if __name__ == '__main__':
    tf_ini()
    mkdiring(args.outdir)
    img=img2np(ffzk(args.train),64)
    gans=gan()
    gans.train(img,epoch=args.epoch,batch=args.batch,predbatch=args.predbatch)
    
    
    
    
    
    
    