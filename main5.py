#functionalAPI_test
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
from tqdm import tqdm
import argparse
import os


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

class bottleneck():
    def __init__(self,dim=4):#plz define used layers below...
        self.dim=dim
        return
    def __call__(self,mod):#plz add layers below...
        with tf.name_scope("B_N"):
            mod_1=mod
            if mod.shape[-1]!=self.dim:
                mod=Conv2D(self.dim,1,padding="same")(mod)
            mod_1=Conv2D(int(self.dim/4),1,padding="same",activation="relu")(mod_1)
            mod_1=Conv2D(int(self.dim/4),3,padding="same",activation="relu")(mod_1)
            mod_1=Conv2D(self.dim,1,padding="same")(mod_1)
            mod=keras.layers.add([mod,mod_1])
            mod=Activation("relu")(mod)
        return mod    
    
def m_gen(dim_rand=8):
    mod_inp = Input(shape=(dim_rand,))
    mod=Reshape((1,1,8))(mod_inp)
    mod=UpSampling2D(2)(mod)
    mod=bottleneck(16)(mod)
    mod=bottleneck(16)(mod)
    mod=UpSampling2D(2)(mod)
    mod=bottleneck(24)(mod)
    mod=bottleneck(24)(mod)
    mod=UpSampling2D(2)(mod)
    mod=bottleneck(32)(mod)
    mod=bottleneck(32)(mod)
    mod=UpSampling2D(2)(mod)
    mod=bottleneck(48)(mod)
    mod=bottleneck(48)(mod)
    mod=bottleneck(48)(mod)
    mod=UpSampling2D(2)(mod)
    mod=bottleneck(64)(mod)
    mod=bottleneck(64)(mod)
    mod=bottleneck(64)(mod)
    mod=UpSampling2D(2)(mod)
    mod=bottleneck(48)(mod)
    mod=bottleneck(48)(mod)
    mod=bottleneck(48)(mod)
    mod=bottleneck(48)(mod)
    mod=Conv2D(3,1,padding="same",activation="sigmoid")(mod)
    return keras.models.Model(inputs=mod_inp, outputs=mod)

def m_dis(dim_rand=8,input_shape=(64,64,3,)):
    mod_inp = Input(shape=input_shape)
    mod=bottleneck(16)(mod_inp)
    mod=bottleneck(16)(mod)
    mod=bottleneck(16)(mod)
    mod=Conv2D(24,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=bottleneck(24)(mod)
    mod=bottleneck(24)(mod)
    mod=bottleneck(24)(mod)
    mod=Conv2D(32,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=bottleneck(32)(mod)
    mod=bottleneck(32)(mod)
    mod=bottleneck(32)(mod)
    mod=Conv2D(48,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=bottleneck(48)(mod)
    mod=bottleneck(48)(mod)
    mod=Conv2D(64,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=bottleneck(64)(mod)
    mod=bottleneck(64)(mod)
    mod=Conv2D(84,4,2,padding="same",activation="relu")(mod)
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
    
def gen_mod():
  mod_inp = tf.keras.layers.Input(shape=(28,28))
  mod=tf.keras.layers.Flatten()(mod_inp)
  mod=tf.keras.layers.Dense(128,activation=tf.nn.relu)(mod)
  mod=tf.keras.layers.Dense(128,activation=tf.nn.relu)(mod)
  mod=tf.keras.layers.Dense(10,activation=tf.nn.relu)(mod)
  return tf.keras.models.Model(inputs=mod_inp, outputs=mod)

class gan():
    def __init__(self,trials=[]):
        self.gen=m_gen(dim_rand=8)
        self.dis=m_dis(dim_rand=8)
    def pred(self,batch=4):
        return self.gen(np.random.rand(batch,8).astype(np.float32))
    def train(self,data=[],epoch=1000,batch=16,predbatch=8):
        
        optimizer = keras.optimizers.SGD(0.003)
        self.gen.compile(optimizer = optimizer,
                          loss=keras.losses.binary_crossentropy)
        self.dis.compile(optimizer = optimizer,
                          loss=keras.losses.binary_crossentropy)
        self.gen.summary()
        self.dis.summary()
        
        try:self.gen.load_weights(os.path.join(args.outdir,"disw.hdf5"))
        except:print("\nCannot_use_savedata...")
        try:self.dis.load_weights(os.path.join(args.outdir,"genw.hdf5"))
        except:print("\nCannot_use_savedata...")
                
        ones=np.ones(batch).astype(np.float32)        
        zeros=np.zeros(batch).astype(np.float32)
        labels=np.random.rand(data.shape[0],8).astype(np.float32)
        
        
        for i in range(epoch):
            with tqdm(total=data.shape[0]) as pbar:
                while pbar.n+batch<pbar.total:
                    datum=data[pbar.n:pbar.n+batch];label=labels[pbar.n:pbar.n+batch];
                
                    with tf.GradientTape() as tape:
                        dis=self.dis(self.gen(label))
                        dis=keras.losses.binary_crossentropy(zeros,dis)
                        dis=tf.reduce_sum(dis) / batch
                        grad=tape.gradient(dis,self.dis.trainable_variables)
                        grad,_ = tf.clip_by_global_norm(grad, 15)
                        optimizer.apply_gradients(zip(grad,self.dis.trainable_variables))
                        del tape
                    
                    with tf.GradientTape() as tape:
                        dis=self.dis(datum)
                        dis=keras.losses.binary_crossentropy(ones,dis)
                        dis=tf.reduce_sum(dis) / batch
                        grad=tape.gradient(dis,self.dis.trainable_variables)
                        grad,_ = tf.clip_by_global_norm(grad, 15)
                        optimizer.apply_gradients(zip(grad,self.dis.trainable_variables))
                        del tape
                    
                    with tf.GradientTape() as tape:
                        dis_gen=self.dis(self.gen(label))
                        dis_gen=keras.losses.binary_crossentropy(ones,dis_gen)
                        dis_gen=tf.reduce_sum(dis_gen) / batch
                        grad=tape.gradient(dis_gen,self.gen.trainable_variables) 
                        grad,_ = tf.clip_by_global_norm(grad, 15)
                        optimizer.apply_gradients(zip(grad,self.gen.trainable_variables))
                        del tape
                    
                    pbar.update(batch)
                    
            dis1=self.dis(self.gen(labels[:batch]))
            dis2=self.dis(data[:batch])
            dis1=tf.reduce_sum(keras.losses.binary_crossentropy (zeros,dis1)) / batch
            dis2=tf.reduce_sum(keras.losses.binary_crossentropy (ones,dis2)) / batch
            print("\nke(fake,tru):",dis1.numpy(),dis2.numpy())
            tf2img(self.pred(predbatch),os.path.join(args.outdir,"1"),epoch=i,ext=".png")
                    
            self.dis.save_weights(os.path.join(args.outdir,"disw.hdf5"))
            self.gen.save_weights(os.path.join(args.outdir,"genw.hdf5"))
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
    
    
    
    
    
    
    