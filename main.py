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

class c3c(keras.Model):
    def __init__(self,dim=4):#plz define used layers below...
        super().__init__()
        self.layer1=[Conv2D(dim,1,padding="same"),
                ]
        self.layer1_1=[Conv2D(dim,3,padding="same"),
                       Conv2D(dim,3,padding="same"),
                       Conv2D(dim,3,padding="same"),
                ]
        self.dim=dim
        return
    def call(self,mod):#plz add layers below...
        with tf.name_scope("c3c"):
            mod_1=mod
            for i in range(len(self.layer1)):mod=self.layer1[i](mod)
            for i in range(len(self.layer1_1)):mod_1=self.layer1_1[i](mod_1)
            mod=keras.layers.add([mod,mod_1])
            mod=Dropout(0.05)(mod)
            mod=Activation("relu")(mod)
        return mod
    
class m_gen(keras.Model):
    def __init__(self,trials=[]):#plz define used layers below...
        super().__init__()        
        self.layer1=[Reshape((1,1,8)),   
                     UpSampling2D(2),
                     c3c(12),
                     UpSampling2D(2),
                     c3c(16),
                     UpSampling2D(2),
                     c3c(24),
                     c3c(32),
                     UpSampling2D(2),
                     c3c(32),
                     c3c(48),
                     UpSampling2D(2),
                     c3c(48),
                     c3c(54),
                     c3c(64),
                     UpSampling2D(2),
                     c3c(64),
                     c3c(72),
                     c3c(84),
                     c3c(94),
                     Conv2D(3,1,padding="same",activation="sigmoid"),
                     ]
        return 
    def call(self, mod,training=None):#plz add layers below...
        for i in range(len(self.layer1)):
            mod=self.layer1[i](mod)
        return mod
    
    
class m_dis(keras.Model):
    def __init__(self):#plz define used layers below...
        super().__init__()        
        self.layer1=[c3c(16),
                     c3c(18),
                     c3c(24),
                     Conv2D(24,4,2,padding="same",activation="relu"),
                     LayerNormalization(),
                     c3c(28),
                     c3c(32),
                     c3c(36),
                     Conv2D(36,4,2,padding="same",activation="relu"),
                     LayerNormalization(),
                     c3c(42),
                     c3c(46),
                     c3c(48),
                     Conv2D(48,4,2,padding="same",activation="relu"),
                     LayerNormalization(),
                     c3c(54),
                     c3c(64),
                     Conv2D(64,4,2,padding="same",activation="relu"),
                     LayerNormalization(),
                     c3c(84),
                     c3c(84),
                     Conv2D(84,4,2,padding="same",activation="relu"),
                     c3c(84),
                     LayerNormalization(),
                     Dropout(0.05),
                     Flatten(),
                     Dense(84,activation="relu"),
                     Dense(64,activation="relu"),
                     LayerNormalization(),
                     Dropout(0.05),
                     Dense(32,activation="relu"),
                     Dense(1,activation="sigmoid"),
                ]
        return 
    def call(self, mod,training=None):#plz add layers below...
        for i in range(len(self.layer1)):
            mod=self.layer1[i](mod)
        return mod
    
def baches(a,b):
    a.reshape((b,a.shape()))
    
class gan():
    def __init__(self,trials=[]):
        self.gen=m_gen()
        self.dis=m_dis()
    def pred(self,batch=4):
        return self.gen(np.random.rand(batch,8).astype(np.float32))
    def train(self,data=[],epoch=1000,batch=16,predbatch=8):
        #unco
        self.gen.build(input_shape=(batch,8))
        self.dis.build(input_shape=(batch,data.shape[1],data.shape[2],data.shape[3]))
        self.gen.summary()
        self.dis.summary()
        
        optimizer = keras.optimizers.SGD(0.003)
        self.gen.compile(optimizer = optimizer,
                          loss=keras.losses.binary_crossentropy,metrics=['accuracy'])
        self.dis.compile(optimizer = optimizer,
                          loss=keras.losses.binary_crossentropy,metrics=['accuracy'])
                
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
        

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train' ,help="train_data",default="./lfw")
parser.add_argument('-o', '--outdir' ,help="outdir",default="./output")
parser.add_argument('-b', '--batch' ,help="batch",default=32,type=int)
parser.add_argument('-p', '--predbatch' ,help="batch_size_of_prediction",default=8,type=int)
parser.add_argument('-e', '--epoch' ,help="epochs",default=30,type=int)
args = parser.parse_args()

if __name__ == '__main__':
    tf_ini()
    mkdiring(args.outdir)
    img=img2np(ffzk(args.train),64)
    gans=gan()
    gans.train(img,epoch=args.epoch,batch=args.batch,predbatch=args.predbatch)
    
    
    
    
    
    
    