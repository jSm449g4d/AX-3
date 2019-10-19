import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
from tqdm import tqdm
import random

from ARutil import ffzk,mkdiring,rootYrel


def img2np(dir=[],img_len=0):
    img=[]
    for x in dir:
        try:img.append(cv2.imread(x))
        except:continue
        if img_len!=0:img[-1]=cv2.resize(img[-1],(img_len,img_len))
        elif img[-1].shape!=img[0].shape:img.pop(-1);continue#Leave only the same shape
        img[-1] = img[-1].astype(np.float32)/ 256
    return np.stack(img, axis=0)

def tf2img(tfs,dir="./",epoch=0,ext=".png"):
    mkdiring(dir)
    tfs=(tfs.numpy()*256).astype(np.uint8)
    for i in range(tfs.shape[0]):
        cv2.imwrite(rootYrel(dir,"epoch-num"+str(epoch)+"-"+str(i)+ext),tfs[i])
        
def tf_ini():#About GPU resources
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
    if len(physical_devices)==0:print("GPU failed!")
    return len(physical_devices)
    
class c3c(keras.Model):
    def __init__(self,dim=4):#plz define used layers below...
        super().__init__()
        self.layer1_1=[Activation("elu"),
                       Conv2D(dim,3,padding="same"),
                       Conv2D(dim,3,padding="same"),
                       Conv2D(dim,3,padding="same"),
                       Dropout(0.05),
                ]
        self.layer1=[Conv2D(dim,1,padding="same"),
                ]
        return
    @tf.function
    def call(self,mod):#plz add layers below...
        mod_1=mod
        for i in range(len(self.layer1_1)):mod_1=self.layer1_1[i](mod_1)
        for i in range(len(self.layer1)):mod=self.layer1[i](mod)
        mod=keras.layers.add([mod,mod_1])
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
    @tf.function
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
                     Conv2D(24,4,2,padding="same",activation="elu"),
                     LayerNormalization(),
                     Dropout(0.05),
                     c3c(28),
                     c3c(32),
                     c3c(36),
                     Conv2D(36,4,2,padding="same",activation="elu"),
                     LayerNormalization(),
                     Dropout(0.05),
                     c3c(42),
                     c3c(46),
                     c3c(48),
                     Conv2D(48,4,2,padding="same",activation="elu"),
                     LayerNormalization(),
                     Dropout(0.05),
                     c3c(54),
                     c3c(64),
                     Conv2D(64,4,2,padding="same",activation="elu"),
                     LayerNormalization(),
                     Dropout(0.05),
                     c3c(84),
                     c3c(84),
                     Conv2D(84,4,2,padding="same",activation="elu"),
                     LayerNormalization(),
                     Dropout(0.05),
                     c3c(84),
                     Flatten(),
                     Dense(84,activation="elu"),
                     Dense(64,activation="elu"),
                     LayerNormalization(),
                     Dense(32,activation="elu"),
                     Dense(1,activation="sigmoid"),
                ]
        return 
    @tf.function
    def call(self, mod,training=None):#plz add layers below...
        for i in range(len(self.layer1)):
            mod=self.layer1[i](mod)
        return mod
    
class gan():
    def __init__(self,trials=[]):
        self.gen=m_gen()
        self.dis=m_dis()
    def pred(self,batch=16):
        return self.gen(np.random.rand(batch,8).astype(np.float32))
    def train(self,data=[],epoch=1000,batch=16):
        optimizer = keras.optimizers.SGD(0.003)
        ones=np.ones(batch).astype(np.float32)
        zeros=np.zeros(batch).astype(np.float32)
        labels=np.random.rand(data.shape[0],8).astype(np.float32)
        
        for i in tqdm(range(epoch)):
            
            ii=random.randint(0,data.shape[0]-batch)
            
            with tf.GradientTape() as tape:#data[:batch]
                dis=self.dis(self.gen(labels[ii:ii+batch]))
                dis=keras.losses.binary_crossentropy(zeros,dis)
                dis=tf.reduce_sum(dis) / batch
                grad=tape.gradient(dis,self.dis.trainable_variables)
                grad,_ = tf.clip_by_global_norm(grad, 15)
                optimizer.apply_gradients(zip(grad,self.dis.trainable_variables))
                del tape
                
            with tf.GradientTape() as tape:
                dis=self.dis(data[ii:ii+batch])
                dis=keras.losses.binary_crossentropy(ones,dis)
                dis=tf.reduce_sum(dis) / batch
                grad=tape.gradient(dis,self.dis.trainable_variables)
                grad,_ = tf.clip_by_global_norm(grad, 15)
                optimizer.apply_gradients(zip(grad,self.dis.trainable_variables))
                del tape
                
            with tf.GradientTape() as tape:
                dis_gen=self.dis(self.gen(labels[ii:ii+batch]))
                dis_gen=keras.losses.binary_crossentropy(ones,dis_gen)
                dis_gen=tf.reduce_sum(dis_gen) / batch
                grad=tape.gradient(dis_gen,self.gen.trainable_variables) 
                grad,_ = tf.clip_by_global_norm(grad, 15)
                optimizer.apply_gradients(zip(grad,self.gen.trainable_variables))#
                del tape
                
            if i%250==0:
                dis1=self.dis(self.gen(labels[ii:ii+batch]))
                dis2=self.dis(data[ii:ii+batch])
                print("e:",dis1,dis2)
                dis1=tf.reduce_sum(keras.losses.binary_crossentropy (zeros,dis1)) / batch
                dis2=tf.reduce_sum(keras.losses.binary_crossentropy (ones,dis2)) / batch
                print("ke:",dis1.numpy(),dis2.numpy())
                tf2img(self.pred(),"./output",i,ext=".png")
                
        dis1=self.dis(self.gen(labels[ii:ii+batch]))
        dis1=tf.reduce_sum(keras.losses.binary_crossentropy (zeros,dis1)) / batch
        return dis1.numpy()
    
if __name__ == '__main__':
    tf_ini()
    img=img2np(ffzk("./apple2orange/trainA"),64)
    gans=gan()
    gans.train(img,epoch=20000,batch=16)
    
    
    
    
    
    