
import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image 
import numpy as np
from random import shuffle
from scipy.misc import imread, imresize
from timeit import default_timer as timer
import time
import sys
sys.path.append("..")
from ssd_utils import BBoxUtility
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import pickle


#****************************************************************************

from keras import backend as t
t.set_image_data_format('channels_last')

from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers import Merge
#from keras.layers.merge import Average
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model, np_utils
from sklearn.utils import shuffle
from keras.layers.core import Reshape
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import ConvLSTM2D
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.layers import Embedding, Masking
from keras import callbacks

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#***********************************************************jadid

class VideoTest(object):
 
     #intial kardane bazi parametrha dar object detection
    def __init__(self, class_names, model, input_shape):
        self.class_names = class_names                          #moshakhas kardane tedade kelas ha
        self.num_classes = len(class_names)
        self.model = model
        self.input_shape = input_shape
        self.bbox_util = BBoxUtility(self.num_classes)
                                                                # Create unique and somewhat visually distinguishable bright
                                                                # colors for the different classes.
        self.class_colors = []                                  #moshakhas kardane har kelas bar hasbe rang
        for i in range(0, self.num_classes):
                                                                # This can probably be written in a more elegant manner
            hue = 255*i/self.num_classes
            col = np.zeros((1,1,3)).astype("uint8")
            col[0][0][0] = hue
            col[0][0][1] = 128 # Saturation
            col[0][0][2] = 255 # Value
            cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
            col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
            self.class_colors.append(col) 
    


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def run(self, video_path = 0, start_frame = 0, conf_thresh = 0.6):
        self.data=[]
        
        def Normalize(data):
            mean_data=np.mean(data)
            std_data=np.std(data)
            norm_data=(data-mean_data)/std_data
            return norm_data
         #********************************************** 
            
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        w=100
        h=100                                                                   #w,h tool o arze frame ra moshakhas mikonad ke mikhahim be shabake emal konim.
        batch_size = 1
        num_channel=1                                                          #batch size yani dar har tekrar shabake chand sample az data ra bbinad ta deghat dahad.
        nb_filters = [20,40,80,160,320,640]                                     #tedade filterhayie layiehayie convolution ra moshakhas mikonad.
        num_classes = 4                                                         #tedade kelass hayie khoroji ma ast.
        
        
        
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        model21 = Sequential() #noe sakhtene modele keras ra taein mikonad.
        model21.add(Masking(mask_value=0, batch_input_shape=(batch_size,None,w,h,1)))               #nekteye mohem in ast ke dar lstm bayad data ra mask konim ta shabake nesbat be toole video mahdood nabashad. vaghti mask mikonim yani aghar yek video daraye toole kamtaz az sayer video ha bood khode shabake b ezaye frame hayie kamtar sefr mikarad.
        #baraye afzodane laye convolution be lstm bayad "timedistributed" estefade konim. parameter hayie in laye niz manande laye convolution mibashad ke dar jaye digari tozih dade shode ast.       
        model21.add(TimeDistributed(Conv2D(nb_filters[0], kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=(1,1),
                                         use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros'), batch_input_shape=(batch_size,None,w,h,1)))
        
        
        model21.add(Activation('relu'))                                                             #be har laye convolution bayad tabe faal konnande ezafe konim. 
        model21.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))  #laye pooling ham bayad "timedistributed" bashad ta be lstm betavan ezafe kard.
        model21.add(Activation('relu'))
        
        
        #sayere layeha mannade layehayie bala hastand va faghat filter convolution tafavot darad.
        model21.add(TimeDistributed(Conv2D(nb_filters[1], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #mode2l.add(TimeDistributed(BatchNormalization()))
        model21.add(Activation('relu'))
        model21.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model21.add(Activation('relu'))
        
        
        
        model21.add(TimeDistributed(Conv2D(nb_filters[2], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model21.add(Activation('relu'))
        model21.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model21.add(Activation('relu'))
        
        
        
        model21.add(TimeDistributed(Conv2D(nb_filters[3], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model21.add(Activation('relu'))
        model21.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model21.add(Activation('relu'))
        
        
        
        
        model21.add(TimeDistributed(Conv2D(nb_filters[4], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #mode2l.add(TimeDistributed(BatchNormalization()))
        model21.add(Activation('relu'))
        model21.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model21.add(Activation('relu'))
        
        
        
        
        model21.add(TimeDistributed(Flatten()))                                  #data ghab az ezafe shodab lstm bayad flat shavad
        model21.add(Masking(mask_value=0, input_shape=model21.output_shape))      #ba in dastor dataye mask shode ra ba meghdare 0 por mikonim ta b lstm bedahim
        model21.add(Bidirectional(LSTM(512, stateful=True)))                     #lstm ra az noe bidirectional entekhab kardim ba tedade norone 128, stateful yani az halate batch feli baraye laye badi estefade konad.
        model21.add(Dropout(0.2))                                                #20% data ra jahate jelogiri az overfitting shabake.
        model21.add(Activation('relu'))
        #model.summary()
        model21.add(Dense(num_classes, activation='softmax'))
        model21.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])
        
        
        #**********************************************************************
        fname1= "/home/tech/works/phase 2/lstm/weight5/Best-weights-my-model-020-0.9934-0.8947.hdf5"
        model21.load_weights(fname1)  #load kardane vaznhayie zakhire shode dar shabake bala
        #**********************************************************************

        
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # hala b haman sorat shabake haye badi ra baraye afrad 2,3,4, ... misazim .
        model22 = Sequential()
        model22.add(Masking(mask_value=0, batch_input_shape=(batch_size,None,w,h,1)))
        model22.add(TimeDistributed(Conv2D(nb_filters[0], kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=(1,1),
                                         use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros'), batch_input_shape=(batch_size,None,w,h,1)))
        model22.add(Activation('relu'))
        model22.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model22.add(Activation('relu'))
        
        
        model22.add(TimeDistributed(Conv2D(nb_filters[1], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model22.add(Activation('relu'))
        model22.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model22.add(Activation('relu'))
        
        
        model22.add(TimeDistributed(Conv2D(nb_filters[2], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model22.add(Activation('relu'))
        model22.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model22.add(Activation('relu'))
        
        
        model22.add(TimeDistributed(Conv2D(nb_filters[3], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model22.add(Activation('relu'))
        model22.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model22.add(Activation('relu'))
        
        
        model22.add(TimeDistributed(Conv2D(nb_filters[4], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model22.add(Activation('relu'))
        model22.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model22.add(Activation('relu'))


        model22.add(TimeDistributed(Flatten()))
        model22.add(Masking(mask_value=0, input_shape=model22.output_shape))
        model22.add(Bidirectional(LSTM(512, stateful=True)))
        model22.add(Dropout(0.2))
        model22.add(Activation('relu'))
        #model2.summary()
        model22.add(Dense(num_classes, activation='softmax'))
        model22.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])
        fname2= "/home/tech/works/phase 2/lstm/weight5/Best-weights-my-model-020-0.9934-0.8947.hdf5"
        model22.load_weights(fname2)
        
        
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        model23 = Sequential()
        model23.add(TimeDistributed(Conv2D(nb_filters[0], kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=(1,1),
                                         use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros'), batch_input_shape=(batch_size,None,w,h,1)))
        model23.add(Activation('relu'))
        model23.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model23.add(Activation('relu'))
        
        
        model23.add(TimeDistributed(Conv2D(nb_filters[1], kernel_size=(3, 3), strides=(1,1), padding='same')))
        model23.add(Activation('relu'))
        model23.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model23.add(Activation('relu'))
        
        
        model23.add(TimeDistributed(Conv2D(nb_filters[2], kernel_size=(3, 3), strides=(1,1), padding='same')))
        model23.add(Activation('relu'))
        model23.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model23.add(Activation('relu'))


        model23.add(TimeDistributed(Conv2D(nb_filters[3], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model23.add(Activation('relu'))
        model23.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model23.add(Activation('relu'))
        
        
        model23.add(TimeDistributed(Conv2D(nb_filters[4], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model23.add(Activation('relu'))
        model23.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model23.add(Activation('relu'))
        
        
        model23.add(TimeDistributed(Flatten()))
        model23.add(Masking(mask_value=0, input_shape=model23.output_shape))
        model23.add(Bidirectional(LSTM(512, stateful=True)))
        model23.add(Dropout(0.2))
        model23.add(Activation('relu'))
        model23.add(Dense(num_classes, activation='softmax'))
        model23.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])
        fname3 = "/home/tech/works/phase 2/lstm/weight5/Best-weights-my-model-020-0.9934-0.8947.hdf5"
        model23.load_weights(fname3)
        
        
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        model24 = Sequential()
        model24.add(Masking(mask_value=0, batch_input_shape=(batch_size,None,w,h,1)))
        model24.add(TimeDistributed(Conv2D(nb_filters[0], kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=(1,1),
                                         use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros'), batch_input_shape=(batch_size,None,w,h,1)))
        model24.add(Activation('relu'))
        model24.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model24.add(Activation('relu'))
        
        
        model24.add(TimeDistributed(Conv2D(nb_filters[1], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model24.add(Activation('relu'))
        model24.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model24.add(Activation('relu'))
        
        
        model24.add(TimeDistributed(Conv2D(nb_filters[2], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model24.add(Activation('relu'))
        model24.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model24.add(Activation('relu'))
        
        
        model24.add(TimeDistributed(Conv2D(nb_filters[3], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model24.add(Activation('relu'))
        model24.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model24.add(Activation('relu'))
        
        
        model24.add(TimeDistributed(Conv2D(nb_filters[4], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model24.add(Activation('relu'))
        model24.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model24.add(Activation('relu'))
        
        
        model24.add(TimeDistributed(Flatten()))
        model24.add(Masking(mask_value=0, input_shape=model24.output_shape))
        model24.add(Bidirectional(LSTM(512, stateful=True)))
        model24.add(Dropout(0.2))
        model24.add(Activation('relu'))
        model24.add(Dense(num_classes, activation='softmax'))
        model24.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])
        fname4= "/home/tech/works/phase 2/lstm/weight5/Best-weights-my-model-020-0.9934-0.8947.hdf5"
        model24.load_weights(fname4)
        
        
        
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        model25 = Sequential()
        model25.add(Masking(mask_value=0, batch_input_shape=(batch_size,None,w,h,1)))
        model25.add(TimeDistributed(Conv2D(nb_filters[0], kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=(1,1),
                                         use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros'), batch_input_shape=(batch_size,None,w,h,1)))
        model25.add(Activation('relu'))
        model25.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model25.add(Activation('relu'))
        
        
        model25.add(TimeDistributed(Conv2D(nb_filters[1], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model25.add(Activation('relu'))
        model25.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model25.add(Activation('relu'))
        
        
        model25.add(TimeDistributed(Conv2D(nb_filters[2], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model25.add(Activation('relu'))
        model25.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model25.add(Activation('relu'))
        
        
        model25.add(TimeDistributed(Conv2D(nb_filters[3], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model25.add(Activation('relu'))
        model25.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model25.add(Activation('relu'))
        
        
        model25.add(TimeDistributed(Conv2D(nb_filters[4], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model25.add(Activation('relu'))
        model25.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model25.add(Activation('relu'))
        
        
        model25.add(TimeDistributed(Flatten()))
        model25.add(Masking(mask_value=0, input_shape=model25.output_shape))
        model25.add(Bidirectional(LSTM(512, stateful=True)))
        model25.add(Dropout(0.2))
        model25.add(Activation('relu'))
        model25.add(Dense(num_classes, activation='softmax'))
        model25.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])
        fname5= "/home/tech/works/phase 2/lstm/weight5/Best-weights-my-model-020-0.9934-0.8947.hdf5"
        model25.load_weights(fname5)        
        
        
        
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        model26 = Sequential()
        model26.add(TimeDistributed(Conv2D(nb_filters[0], kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=(1,1),
                                         use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros'), batch_input_shape=(batch_size,None,w,h,1)))
        model26.add(Activation('relu'))
        model26.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model26.add(Activation('relu'))
        
        
        model26.add(TimeDistributed(Conv2D(nb_filters[1], kernel_size=(3, 3), strides=(1,1), padding='same')))
        model26.add(Activation('relu'))
        model26.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model26.add(Activation('relu'))
        
        
        model26.add(TimeDistributed(Conv2D(nb_filters[2], kernel_size=(3, 3), strides=(1,1), padding='same')))
        model26.add(Activation('relu'))
        model26.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model26.add(Activation('relu'))


        model26.add(TimeDistributed(Conv2D(nb_filters[3], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model26.add(Activation('relu'))
        model26.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model26.add(Activation('relu'))
        
        
        model26.add(TimeDistributed(Conv2D(nb_filters[4], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model26.add(Activation('relu'))
        model26.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model26.add(Activation('relu'))
        
        
        model26.add(TimeDistributed(Flatten()))
        model26.add(Masking(mask_value=0, input_shape=model26.output_shape))
        model26.add(Bidirectional(LSTM(512, stateful=True)))
        model26.add(Dropout(0.2))
        model26.add(Activation('relu'))
        model26.add(Dense(num_classes, activation='softmax'))
        model26.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])
        fname6 = "/home/tech/works/phase 2/lstm/weight5/Best-weights-my-model-020-0.9934-0.8947.hdf5"
        model26.load_weights(fname6)
        
        
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        model27 = Sequential()
        model27.add(Masking(mask_value=0, batch_input_shape=(batch_size,None,w,h,1)))
        model27.add(TimeDistributed(Conv2D(nb_filters[0], kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=(1,1),
                                         use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros'), batch_input_shape=(batch_size,None,w,h,1)))
        model27.add(Activation('relu'))
        model27.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model27.add(Activation('relu'))
        
        
        model27.add(TimeDistributed(Conv2D(nb_filters[1], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model27.add(Activation('relu'))
        model27.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model27.add(Activation('relu'))
        
        
        model27.add(TimeDistributed(Conv2D(nb_filters[2], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model27.add(Activation('relu'))
        model27.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model27.add(Activation('relu'))
        
        
        model27.add(TimeDistributed(Conv2D(nb_filters[3], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model27.add(Activation('relu'))
        model27.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model27.add(Activation('relu'))
        
        
        model27.add(TimeDistributed(Conv2D(nb_filters[4], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model27.add(Activation('relu'))
        model27.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model27.add(Activation('relu'))
        
        
        model27.add(TimeDistributed(Flatten()))
        model27.add(Masking(mask_value=0, input_shape=model27.output_shape))
        model27.add(Bidirectional(LSTM(512, stateful=True)))
        model27.add(Dropout(0.2))
        model27.add(Activation('relu'))
        model27.add(Dense(num_classes, activation='softmax'))
        model27.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])
        fname7= "/home/tech/works/phase 2/lstm/weight5/Best-weights-my-model-020-0.9934-0.8947.hdf5"
        model27.load_weights(fname7)
        
        
        
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        model28 = Sequential()
        model28.add(Masking(mask_value=0, batch_input_shape=(batch_size,None,w,h,1)))
        model28.add(TimeDistributed(Conv2D(nb_filters[0], kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=(1,1),
                                         use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros'), batch_input_shape=(batch_size,None,w,h,1)))
        model28.add(Activation('relu'))
        model28.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model28.add(Activation('relu'))
        
        
        model28.add(TimeDistributed(Conv2D(nb_filters[1], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model28.add(Activation('relu'))
        model28.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model28.add(Activation('relu'))
        
        
        model28.add(TimeDistributed(Conv2D(nb_filters[2], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model28.add(Activation('relu'))
        model28.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model28.add(Activation('relu'))
        
        
        model28.add(TimeDistributed(Conv2D(nb_filters[3], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model28.add(Activation('relu'))
        model28.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model28.add(Activation('relu'))
        
        
        model28.add(TimeDistributed(Conv2D(nb_filters[4], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model28.add(Activation('relu'))
        model28.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model28.add(Activation('relu'))
        
        
        model28.add(TimeDistributed(Flatten()))
        model28.add(Masking(mask_value=0, input_shape=model28.output_shape))
        model28.add(Bidirectional(LSTM(512, stateful=True)))
        model28.add(Dropout(0.2))
        model28.add(Activation('relu'))
        model28.add(Dense(num_classes, activation='softmax'))
        model28.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])
        fname8= "/home/tech/works/phase 2/lstm/weight5/Best-weights-my-model-020-0.9934-0.8947.hdf5"
        model28.load_weights(fname8)
        
        
        
        
        
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        model29 = Sequential()
        model29.add(Masking(mask_value=0, batch_input_shape=(batch_size,None,w,h,1)))
        model29.add(TimeDistributed(Conv2D(nb_filters[0], kernel_size=(3, 3), strides=(1,1), padding='same',dilation_rate=(1,1),
                                         use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros'), batch_input_shape=(batch_size,None,w,h,1)))
        model29.add(Activation('relu'))
        model29.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model29.add(Activation('relu'))
        
        
        model29.add(TimeDistributed(Conv2D(nb_filters[1], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model29.add(Activation('relu'))
        model29.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model29.add(Activation('relu'))
        
        
        model29.add(TimeDistributed(Conv2D(nb_filters[2], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model29.add(Activation('relu'))
        model29.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model29.add(Activation('relu'))
        
        
        model29.add(TimeDistributed(Conv2D(nb_filters[3], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model29.add(Activation('relu'))
        model29.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model29.add(Activation('relu'))
        
        
        model29.add(TimeDistributed(Conv2D(nb_filters[4], kernel_size=(3, 3), strides=(1,1), padding='same')))
        #model.add(TimeDistributed(BatchNormalization()))
        model29.add(Activation('relu'))
        model29.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same',strides=(2,2))))
        model29.add(Activation('relu'))
        
        
        model29.add(TimeDistributed(Flatten()))
        model29.add(Masking(mask_value=0, input_shape=model28.output_shape))
        model29.add(Bidirectional(LSTM(512, stateful=True)))
        model29.add(Dropout(0.2))
        model29.add(Activation('relu'))
        model29.add(Dense(num_classes, activation='softmax'))
        model29.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])
        fname9= "/home/tech/works/phase 2/lstm/weight5/Best-weights-my-model-020-0.9934-0.8947.hdf5"
        model29.load_weights(fname8)
        

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #*********************************************************jadid
        #load kardane model m label haei k zakhire kardim
        model1 = load_model('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/weight')
        mlb = pickle.loads(open('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/lb', "rb").read())


        model2 = load_model('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/weight')
        mlb = pickle.loads(open('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/lb', "rb").read())
        
        model3 = load_model('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/weight')
        mlb = pickle.loads(open('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/lb', "rb").read())
        
        model4 = load_model('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/weight')
        mlb = pickle.loads(open('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/lb', "rb").read())
        
        model5 = load_model('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/weight')
        mlb = pickle.loads(open('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/lb', "rb").read())
        
        
        model5 = load_model('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/weight')
        mlb = pickle.loads(open('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/lb', "rb").read())
        
        
        model6 = load_model('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/weight')
        mlb = pickle.loads(open('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/lb', "rb").read())
        
        
        model7 = load_model('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/weight')
        mlb = pickle.loads(open('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/lb', "rb").read())
        
        
        model8 = load_model('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/weight')
        mlb = pickle.loads(open('/media/tech/36CAA6EACAA6A619/Action recognition_clothes_database/weight&label/lb', "rb").read())
        
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
        
        
        #*********************************************************************
        person_1 = []
        person_2 = []
        person_3 = []
        person_4 = []
        person_5 = []
        person_6 = []
        person_7 = []
        person_8 = []

        #*********************************************************************
        listdata_frame_person1 = []
        listdata_frame_person2 = []
        listdata_frame_person3 = []
        listdata_frame_person4 = []
        listdata_frame_person5 = []
        listdata_frame_person6 = []
        listdata_frame_person7 = []
        listdata_frame_person8 = []

        
        
        #************************************************************
        vid = cv2.VideoCapture(video_path)#gereftane video
        if not vid.isOpened():
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))
        
        # Compute aspect ratio of video     
        vidw = vid.get(3)                                                #gereftane arze frame
        vidh = vid.get(4)                                                #gereftane toole frame
        vidar = vidw/vidh 
        
        # Skip frames until reaching start_frame
        if start_frame > 0:
            vid.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_frame)
        self.counter2 = 0
        self.counter2_2 = 0
        self.counter2_3 = 0
        self.counter2_4 = 0
        self.counter2_5 = 0
        self.counter2_6 = 0
        self.counter2_7 = 0
        self.counter2_8 = 0

         
         #***********************************************************
        while True:
            retval, orig_image = vid.read()                              #khandane frame ha
            if not retval:
                print("Done!")
                return
                
            im_size = (self.input_shape[0], self.input_shape[1])         #gereftane size frame az tarige shape   
            resized = cv2.resize(orig_image, im_size)                    # taghire size frame
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)               #taghire range be rgb
            
                                                                         # Reshape to original aspect ratio for later visualization
                                                                         # The resized version is used, to visualize what kind of resolution
                                                                         # the network has to work with.
            to_draw = cv2.resize(resized, (int(self.input_shape[0]*vidar), self.input_shape[1])) # taghire size frame b tool va arze hasel az shape
            
            # Use model to predict 
            inputs = [image.img_to_array(rgb)]                           # tabdile vrodi b array
            tmp_inp = np.array(inputs)
            x = preprocess_input(tmp_inp)
            y = self.model.predict(x)                                    #  pishbini khoroji object detection
            
            
            
            #*********************************************************
            # This line creates a new TensorFlow device every time. Is there a 
            results = self.bbox_util.detection_out(y) #bounding box
            
            self.count1=[]
            self.counter = 0
            if len(results) > 0 and len(results[0]) > 0:
                # Interpret output, only one frame is used 
                
                det_label = results[0][:, 0]
                det_conf = results[0][:, 1]
                det_xmin = results[0][:, 2]
                det_ymin = results[0][:, 3]
                det_xmax = results[0][:, 4]
                det_ymax = results[0][:, 5]

                top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]
                


                #******************************************************
                for i in range(top_conf.shape[0]):
                    xmin = int(round(top_xmin[i] * to_draw.shape[1]))
                    ymin = int(round(top_ymin[i] * to_draw.shape[0]))
                    xmax = int(round(top_xmax[i] * to_draw.shape[1]))
                    ymax = int(round(top_ymax[i] * to_draw.shape[0]))

                    # Draw the box on top of the to_draw image
                    class_num = int(top_label_indices[i])
                      
                      
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if self.class_names[class_num]=='person':                         # ba anjam marahele bala agar khoroji kelas "car" bod boro karhaye paein ra anjam bede.
                        self.counter += 1
                        a=cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax-50),        # box keshidan dore mashine
                                  self.class_colors[class_num], 2) 
                        #a3=cv2.rectangle(to_draw, (xmin, ymin+100), (xmax, ymax),        # box keshidan dore mashine
                                  #self.class_colors[class_num], 2)
                        self.count1.append(self.counter)
                        last_item=max(self.count1)
                        text = self.class_names[class_num] + " " + ('%.2f' % top_conf[i])
                        #text_counter=  " " + (num_car[i])
                        text_top = (xmin, ymin-10)                                  # moshakhas kardane mahale text         
                        text_bot = (xmin + 80, ymin + 5) 
                        text_pos = (xmin + 5, ymin)
                        text_pos_counter = (xmin + 5, ymin - 10)
                        cv2.rectangle(to_draw, text_top, text_bot, self.class_colors[class_num], -1)   
                        cv2.putText(to_draw, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1) # put kardane neveshteha bar roye box
                        #cv2.putText(to_draw, text_counter, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                    

                        #*********************************************
                        letter = a[ymin:ymax-50,xmin:xmax]
                        #letter3 = a3[ymin+100:ymax,xmin:xmax]
                        #cv2.imshow('clothes', letter)
                        #cv2.imshow('pants', letter3)
                        
                        letter2 = cv2.resize(letter, (w,h))
                        frame22 = cv2.cvtColor(letter2, cv2.COLOR_RGB2BGR)
                        next = cv2.cvtColor(frame22,cv2.COLOR_BGR2GRAY)
                        
                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        #*********************************************
                        #if num_car[0]=='3':  # agar tedade mashinha 3 ta tashkhis dadi
                            #output = imutils.resize(letter, width=400)
                        frame=cv2.resize(letter,(96,96))                                       # resize kon
                        frame = frame.astype("float") / 255.0                                  # normalize kon
                        input_data=img_to_array(frame)                                         #tabdil be arraye kon
                        input_data = np.expand_dims(input_data, axis=0)
                        print(input_data.shape)
                            
                        # labels 
                        print("[INFO] classifying image...")
                        proba = model1.predict(input_data)[0]                                  #az tarighe model pishbini kon
                        idxs = np.argsort(proba)[::-1][:2]
                        
                         
                        # loop over 
                        for (i, j) in enumerate(idxs):
                                                                                               # build the label and draw the label on the image
                            label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
                            cv2.putText(letter, label, (10, (i * 30) + 25), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)     # pishbini ke kardi bar roye tasvir put kon

                            if mlb.classes_[j]=='blue':
                                self.counter2 += 1
                                person_1.append(next)
                                text_count=  "ID: 1"
                                cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                if self.counter2==5:
                                    input = np.array(person_1)
                                    input = input.astype('float32')
                                    listdata_frame_person1.append(input)
                                    self.counter2=0
                                    test_video = np.array(listdata_frame_person1)
                                    num_of_samples=len(test_video)
                                    print('Number of samples: ', num_of_samples)
                                    print('data_shape: ', test_video.shape)
                                    
                                    if num_channel==1:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.expand_dims(test_video,axis=1)
                                        else:
                                            test_video=np.expand_dims(test_video,axis=4)    
                                    else:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.rollaxis(test_video,3,1)
                                        else:
                                            test_video=np.rollaxis(test_video,3,4)
                                                            
                            #*************************************************     
                                    test_video=Normalize(test_video)
                                    out_kelass = model21.predict_classes(test_video)

                            #*************************************************

                                    text_pos_label = (xmin + 5, ymin - 20)
                                    if out_kelass==[0]:
                                        print('boxing')
                                        cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        
                                    elif out_kelass==[1]:
                                        print('unknown_action')
                                        cv2.putText(to_draw, 'unknown_action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  

                                    elif out_kelass==[2]:
                                        print('walking')
                                        cv2.putText(to_draw, 'walking' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  

                                    elif out_kelass==[3]:
                                        print('waving')
                                        cv2.putText(to_draw, 'waving' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  

                                    person_1 = []
                                    listdata_frame_person1 = []
                                    
                            if mlb.classes_[j]=='brown':
                                self.counter2 += 1
                                person_1.append(next)
                                text_count=  "ID: 2"
                                cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                if self.counter2==5:      
                                    input = np.array(person_1)
                                    #print(input.shape)
                                    input = input.astype('float32')
                                    listdata_frame_person1.append(input)
                                    self.counter2=0
                                    test_video = np.array(listdata_frame_person1)
                                    num_of_samples=len(test_video)
                                    print('Number of samples2: ', num_of_samples)
                                    print('data_shape2: ', test_video.shape)
                                        
                                    if num_channel==1:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.expand_dims(test_video,axis=1)
                                        else:
                                            test_video=np.expand_dims(test_video,axis=4)    
                                    else:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.rollaxis(test_video,3,1)
                                        else:
                                            test_video=np.rollaxis(test_video,3,4)
                                            
                                    #****************************************
                                    print(test_video.shape)                    
                                    test_video=Normalize(test_video)                     
                                    out_kelass2 = model22.predict_classes(test_video)     
                                    text_pos_label = (xmin + 5, ymin - 20)
                                    #****************************************
                                    
                                    if out_kelass2==[0]:
                                        print('boxing')
                                        cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)   
                                        
                                    elif out_kelass2==[1]:
                                        print('unknown_action')
                                        cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
    
                                    elif out_kelass2==[2]:
                                        print('walking')
                                        cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                    elif out_kelass2==[3]:
                                        print('waving')
                                        cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                    person_1 = []
                                    listdata_frame_person1 = []
                                    
                            if mlb.classes_[j]=='green':
                                self.counter2 += 1
                                person_1.append(next)
                                text_count=  "ID: 3"
                                cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                if self.counter2==5:         
                                    input = np.array(person_1)
                                    #print(input.shape)
                                    input = input.astype('float32')
                                    listdata_frame_person1.append(input)
                                    self.counter2=0
                                    test_video = np.array(listdata_frame_person1)
                                    num_of_samples=len(test_video)
                                    print('Number of samples3: ', num_of_samples)
                                    print('data_shape3: ', test_video.shape)
                                        
                                    if num_channel==1:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.expand_dims(test_video,axis=1)
                                        else:
                                            test_video=np.expand_dims(test_video,axis=4)    
                                    else:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.rollaxis(test_video,3,1)
                                        else:
                                            test_video=np.rollaxis(test_video,3,4)
                                            
                                    #*****************************************
                                    print(test_video.shape)                       
                                    test_video=Normalize(test_video)                     
                                    out_kelass3 = model23.predict_classes(test_video)        
                                    text_pos_label = (xmin + 5, ymin - 20)
                                    #****************** **********************                                   
                                    
                                    if out_kelass3==[0]:
                                        print('boxing')
                                        cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                        
                                    elif out_kelass3==[1]:
                                        print('unknown action')
                                        cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
    
                                    elif out_kelass3==[2]:
                                        print('walking')
                                        cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                    elif out_kelass3==[3]:
                                        print('waving')
                                        cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                    person_1 = []
                                    listdata_frame_person1 = []  


                            if mlb.classes_[j]=='orange':
                                self.counter2 += 1
                                person_1.append(next)
                                text_count=  "ID: 4"
                                cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                if self.counter2==5:       
                                    input = np.array(person_1)
                                    #print(input.shape)
                                    input = input.astype('float32')
                                    listdata_frame_person1.append(input)
                                    self.counter2=0
                                    test_video = np.array(listdata_frame_person1)
                                    num_of_samples=len(test_video)
                                    print('Number of samples4: ', num_of_samples)
                                    print('data_shape4: ', test_video.shape)
                                        
                                    if num_channel==1:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.expand_dims(test_video,axis=1)
                                        else:
                                            test_video=np.expand_dims(test_video,axis=4)    
                                    else:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.rollaxis(test_video,3,1)
                                        else:
                                            test_video=np.rollaxis(test_video,3,4)
                                            
                                    #*****************************************
                                    print(test_video.shape)                      
                                    test_video=Normalize(test_video)                
                                    out_kelass4 = model24.predict_classes(test_video)        
                                    text_pos_label = (xmin + 5, ymin - 20)
                                    #*****************************************
                                    
                                    if out_kelass4==[0]:
                                        print('boxing')
                                        cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                        
                                    elif out_kelass4==[1]:
                                        print('unknown action')
                                        cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
    
                                    elif out_kelass4==[2]:
                                        print('walking')
                                        cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                    elif out_kelass4==[3]:
                                        print('waving')
                                        cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                    person_1 = []
                                    listdata_frame_person1 = []
                                
                                
                            if mlb.classes_[j]=='purple':
                                self.counter2 += 1
                                person_1.append(next)
                                text_count=  "ID: 5"
                                cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                if self.counter2==5:        
                                    input = np.array(person_1)
                                    #print(input.shape)
                                    input = input.astype('float32')
                                    listdata_frame_person1.append(input)
                                    self.counter2=0
                                    test_video = np.array(listdata_frame_person1)
                                    num_of_samples=len(test_video)
                                    print('Number of samples5: ', num_of_samples)
                                    print('data_shape5: ', test_video.shape)
                                        
                                    if num_channel==1:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.expand_dims(test_video,axis=1)
                                        else:
                                            test_video=np.expand_dims(test_video,axis=4)    
                                    else:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.rollaxis(test_video,3,1)
                                        else:
                                            test_video=np.rollaxis(test_video,3,4)
                                      
                                    #*****************************************
                                    print(test_video.shape)                      
                                    test_video=Normalize(test_video)                     
                                    out_kelass5 = model25.predict_classes(test_video)         
                                    text_pos_label = (xmin + 5, ymin - 20)
                                    #*****************************************
                                    
                                    if out_kelass5==[0]:
                                        print('boxing')
                                        cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                        
                                    elif out_kelass5==[1]:
                                        print('unknown action')
                                        cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
    
                                    elif out_kelass5==[2]:
                                        print('walking')
                                        cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                    elif out_kelass5==[3]:
                                        print('waving')
                                        cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                    person_1 = []
                                    listdata_frame_person1 = []                 

    
                            if mlb.classes_[j]=='red':
                                self.counter2 += 1
                                person_1.append(next)
                                #self.counter2 += 1
                                #print(self.counter2)
                                text_count=  "ID: 6"
                                cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                if self.counter2==5:        
                                    input = np.array(person_1)
                                    #print(input.shape)
                                    input = input.astype('float32')
                                    listdata_frame_person1.append(input)
                                    self.counter2=0
                                    test_video = np.array(listdata_frame_person1)
                                    num_of_samples=len(test_video)
                                    print('Number of samples5: ', num_of_samples)
                                    print('data_shape5: ', test_video.shape)
                                        
                                    if num_channel==1:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.expand_dims(test_video,axis=1)
                                        else:
                                            test_video=np.expand_dims(test_video,axis=4)    
                                    else:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.rollaxis(test_video,3,1)
                                        else:
                                            test_video=np.rollaxis(test_video,3,4)
                                      
                                    #*****************************************
                                    print(test_video.shape)                      
                                    test_video=Normalize(test_video)                     
                                    out_kelass6 = model26.predict_classes(test_video)         
                                    text_pos_label = (xmin + 5, ymin - 20)
                                    #*****************************************
                                    
                                    if out_kelass6==[0]:
                                        print('boxing')
                                        cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                        
                                    elif out_kelass6==[1]:
                                        print('unknown action')
                                        cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
    
                                    elif out_kelass6==[2]:
                                        print('walking')
                                        cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                    elif out_kelass6==[3]:
                                        print('waving')
                                        cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                    person_1 = []
                                    listdata_frame_person1 = []


                            if mlb.classes_[j]=='white':
                                self.counter2 += 1
                                person_1.append(next)
                                text_count=  "ID: 7"
                                cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                if self.counter2==5:        
                                    input = np.array(person_1)
                                    #print(input.shape)
                                    input = input.astype('float32')
                                    listdata_frame_person1.append(input)
                                    self.counter2=0
                                    test_video = np.array(listdata_frame_person1)
                                    num_of_samples=len(test_video)
                                    print('Number of samples5: ', num_of_samples)
                                    print('data_shape5: ', test_video.shape)
                                        
                                    if num_channel==1:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.expand_dims(test_video,axis=1)
                                        else:
                                            test_video=np.expand_dims(test_video,axis=4)    
                                    else:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.rollaxis(test_video,3,1)
                                        else:
                                            test_video=np.rollaxis(test_video,3,4)
                                      
                                    #*****************************************
                                    print(test_video.shape)                      
                                    test_video=Normalize(test_video)                     
                                    out_kelass7 = model27.predict_classes(test_video)         
                                    text_pos_label = (xmin + 5, ymin - 20)
                                    #*****************************************
                                    
                                    if out_kelass7==[0]:
                                        print('boxing')
                                        cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                        
                                    elif out_kelass7==[1]:
                                        print('unknown action')
                                        cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
    
                                    elif out_kelass7==[2]:
                                        print('walking')
                                        cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                    elif out_kelass7==[3]:
                                        print('waving')
                                        cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                    person_1 = []
                                    listdata_frame_person1 = []
                                
                                
                                
                            if mlb.classes_[j]=='yellow':
                                self.counter2 += 1
                                person_1.append(next)
                                text_count=  "ID: 8"
                                cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                if self.counter2==5:        
                                    input = np.array(person_1)
                                    #print(input.shape)
                                    input = input.astype('float32')
                                    listdata_frame_person1.append(input)
                                    self.counter2=0
                                    test_video = np.array(listdata_frame_person1)
                                    num_of_samples=len(test_video)
                                    print('Number of samples5: ', num_of_samples)
                                    print('data_shape5: ', test_video.shape)
                                        
                                    if num_channel==1:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.expand_dims(test_video,axis=1)
                                        else:
                                            test_video=np.expand_dims(test_video,axis=4)    
                                    else:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.rollaxis(test_video,3,1)
                                        else:
                                            test_video=np.rollaxis(test_video,3,4)
                                      
                                    #*****************************************
                                    print(test_video.shape)                      
                                    test_video=Normalize(test_video)                     
                                    out_kelass8 = model28.predict_classes(test_video)         
                                    text_pos_label = (xmin + 5, ymin - 20)
                                    #*****************************************
                                    
                                    if out_kelass8==[0]:
                                        print('boxing')
                                        cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                        
                                    elif out_kelass8==[1]:
                                        print('unknown action')
                                        cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
    
                                    elif out_kelass8==[2]:
                                        print('walking')
                                        cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                    elif out_kelass8==[3]:
                                        print('waving')
                                        cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                    person_1 = []
                                    listdata_frame_person1 = []
                                    
                                    
                                    
                                    
                            if mlb.classes_[j]=='black':
                                self.counter2 += 1
                                person_1.append(next)
                                text_count=  "ID: 9"
                                cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                if self.counter2==5:        
                                    input = np.array(person_1)
                                    #print(input.shape)
                                    input = input.astype('float32')
                                    listdata_frame_person1.append(input)
                                    self.counter2=0
                                    test_video = np.array(listdata_frame_person1)
                                    num_of_samples=len(test_video)
                                    print('Number of samples5: ', num_of_samples)
                                    print('data_shape5: ', test_video.shape)
                                        
                                    if num_channel==1:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.expand_dims(test_video,axis=1)
                                        else:
                                            test_video=np.expand_dims(test_video,axis=4)    
                                    else:
                                        if t.image_data_format()=='channels_first':
                                            test_video=np.rollaxis(test_video,3,1)
                                        else:
                                            test_video=np.rollaxis(test_video,3,4)
                                      
                                    #*****************************************
                                    print(test_video.shape)                      
                                    test_video=Normalize(test_video)                     
                                    out_kelass66 = model29.predict_classes(test_video)         
                                    text_pos_label = (xmin + 5, ymin - 20)
                                    #*****************************************
                                    
                                    if out_kelass66==[0]:
                                        print('boxing')
                                        cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                        
                                    elif out_kelass66==[1]:
                                        print('unknown action')
                                        cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
    
                                    elif out_kelass66==[2]:
                                        print('walking')
                                        cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                    elif out_kelass66==[3]:
                                        print('waving')
                                        cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                    person_1 = []
                                    listdata_frame_person1 = []
                                
                                   
                                               
                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                               
                        if last_item==2: # agar akharin tedade afrad k tashkhis dadi 2 bod?
        
                                frame2 = cv2.resize(letter,(96,96))
                                frame2 = frame2.astype("float") / 255.0
                                input_data2 = img_to_array(frame2) 
                                input_data2 = np.expand_dims(input_data2, axis=0)
                                
                                proba2 = model2.predict(input_data2)[0]
                                idxs2 = np.argsort(proba2)[::-1][:2]
                            
                                # loop over 
                                for (i, j) in enumerate(idxs2):
                                    # build the label and draw the label on the image
                                    label2 = "{}: {:.2f}%".format(mlb.classes_[j], proba2[j] * 100)
                                    cv2.putText(letter, label2, (385, (i * 30) + 25), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    if mlb.classes_[j]=='blue':
                                        self.counter2_2 += 1
                                        person_2.append(next)
                                        text_count=  "ID: 1"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_2==5:
                                            input2 = np.array(person_2)
                                            input2 = input2.astype('float32')
                                            listdata_frame_person2.append(input2)
                                            self.counter2_2=0
                                            test_video2 = np.array(listdata_frame_person2)
                                            num_of_samples2=len(test_video2)
                                            print('Number of samples: ', num_of_samples2)
                                            print('data_shape: ', test_video2.shape)
                                            
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.expand_dims(test_video2,axis=1)
                                                else:
                                                    test_video2=np.expand_dims(test_video2,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.rollaxis(test_video2,3,1)
                                                else:
                                                    test_video2=np.rollaxis(test_video2,3,4)
                                                                    
                                    #*************************************************     
                                            test_video2=Normalize(test_video2)
                                            out_kelass9 = model21.predict_classes(test_video2)
        
                                    #*************************************************
        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            if out_kelass9==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                                
                                            elif out_kelass9==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown_action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass9==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass9==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            person_2 = []
                                            listdata_frame_person2 = []
                                            
                                    if mlb.classes_[j]=='brown':
                                        self.counter2_2 += 1
                                        person_2.append(next)
                                        text_count=  "ID: 2"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_2==5:      
                                            input2 = np.array(person_2)
                                            #print(input.shape)
                                            input2 = input2.astype('float32')
                                            listdata_frame_person2.append(input2)
                                            self.counter2_2=0
                                            test_video2 = np.array(listdata_frame_person2)
                                            num_of_samples2=len(test_video2)
                                            print('Number of samples2: ', num_of_samples2)
                                            print('data_shape2: ', test_video2.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.expand_dims(test_video2,axis=1)
                                                else:
                                                    test_video2=np.expand_dims(test_video2,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.rollaxis(test_video2,3,1)
                                                else:
                                                    test_video2=np.rollaxis(test_video2,3,4)
                                                    
                                            #****************************************
                                            print(test_video2.shape)                    
                                            test_video2=Normalize(test_video2)                     
                                            out_kelass10 = model22.predict_classes(test_video2)     
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************************************
                                            
                                            if out_kelass10==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)   
                                                
                                            elif out_kelass10==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass10==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass10==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_2 = []
                                            listdata_frame_person2 = []
                                            
                                    if mlb.classes_[j]=='green':
                                        self.counter2_2 += 1
                                        person_2.append(next)
                                        text_count=  "ID: 3"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_2==5:         
                                            input2 = np.array(person_2)
                                            #print(input.shape)
                                            input2 = input2.astype('float32')
                                            listdata_frame_person2.append(input2)
                                            self.counter2_2=0
                                            test_video2 = np.array(listdata_frame_person2)
                                            num_of_samples2=len(test_video2)
                                            print('Number of samples3: ', num_of_samples2)
                                            print('data_shape3: ', test_video2.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.expand_dims(test_video2,axis=1)
                                                else:
                                                    test_video2=np.expand_dims(test_video2,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.rollaxis(test_video2,3,1)
                                                else:
                                                    test_video2=np.rollaxis(test_video2,3,4)
                                                    
                                            #*****************************************
                                            print(test_video2.shape)                       
                                            test_video2=Normalize(test_video2)                     
                                            out_kelass11 = model23.predict_classes(test_video2)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************** **********************                                   
                                            
                                            if out_kelass11==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass11==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass11==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass11==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_2 = []
                                            listdata_frame_person2 = []  
        
        
                                    if mlb.classes_[j]=='orange':
                                        self.counter2_2 += 1
                                        person_2.append(next)
                                        text_count=  "ID: 4"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_2==5:       
                                            input2 = np.array(person_2)
                                            #print(input.shape)
                                            input2 = input2.astype('float32')
                                            listdata_frame_person2.append(input2)
                                            self.counter2_2=0
                                            test_video2 = np.array(listdata_frame_person2)
                                            num_of_samples2=len(test_video2)
                                            print('Number of samples4: ', num_of_samples2)
                                            print('data_shape4: ', test_video2.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.expand_dims(test_video2,axis=1)
                                                else:
                                                    test_video2=np.expand_dims(test_video2,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.rollaxis(test_video2,3,1)
                                                else:
                                                    test_video2=np.rollaxis(test_video2,3,4)
                                                    
                                            #*****************************************
                                            print(test_video2.shape)                      
                                            test_video2=Normalize(test_video2)                
                                            out_kelass12 = model24.predict_classes(test_video2)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass12==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass12==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass12==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass12==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_2 = []
                                            listdata_frame_person2 = []
                                        
                                        
                                    if mlb.classes_[j]=='purple':
                                        self.counter2_2 += 1
                                        person_2.append(next)
                                        text_count=  "ID: 5"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_2==5:        
                                            input2 = np.array(person_2)
                                            #print(input.shape)
                                            input2 = input2.astype('float32')
                                            listdata_frame_person2.append(input2)
                                            self.counter2_2=0
                                            test_video2 = np.array(listdata_frame_person2)
                                            num_of_samples2=len(test_video2)
                                            print('Number of samples5: ', num_of_samples2)
                                            print('data_shape5: ', test_video2.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.expand_dims(test_video2,axis=1)
                                                else:
                                                    test_video2=np.expand_dims(test_video2,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.rollaxis(test_video2,3,1)
                                                else:
                                                    test_video2=np.rollaxis(test_video2,3,4)
                                              
                                            #*****************************************
                                            print(test_video2.shape)                      
                                            test_video2=Normalize(test_video2)                     
                                            out_kelass13 = model25.predict_classes(test_video2)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass13==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass13==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass13==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass13==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_2 = []
                                            listdata_frame_person2 = []                 
        
            
                                    if mlb.classes_[j]=='red':
                                        self.counter2_2 += 1
                                        person_2.append(next)
                                        #self.counter2 += 1
                                        #print(self.counter2)
                                        text_count=  "ID: 6"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_2==5:        
                                            input2 = np.array(person_2)
                                            #print(input.shape)
                                            input2 = input2.astype('float32')
                                            listdata_frame_person2.append(input2)
                                            self.counter2_2=0
                                            test_video2 = np.array(listdata_frame_person2)
                                            num_of_samples2=len(test_video2)
                                            print('Number of samples5: ', num_of_samples2)
                                            print('data_shape5: ', test_video2.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.expand_dims(test_video2,axis=1)
                                                else:
                                                    test_video2=np.expand_dims(test_video2,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.rollaxis(test_video2,3,1)
                                                else:
                                                    test_video2=np.rollaxis(test_video2,3,4)
                                              
                                            #*****************************************
                                            print(test_video2.shape)                      
                                            test_video2=Normalize(test_video2)                     
                                            out_kelass14 = model26.predict_classes(test_video2)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass14==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass14==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass14==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass14==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_2 = []
                                            listdata_frame_person2 = []
        
        
                                    if mlb.classes_[j]=='white':
                                        self.counter2_2 += 1
                                        person_2.append(next)
                                        text_count=  "ID: 7"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_2==5:        
                                            input2 = np.array(person_2)
                                            #print(input.shape)
                                            input2 = input2.astype('float32')
                                            listdata_frame_person2.append(input2)
                                            self.counter2_2=0
                                            test_video2 = np.array(listdata_frame_person2)
                                            num_of_samples2=len(test_video2)
                                            print('Number of samples5: ', num_of_samples2)
                                            print('data_shape5: ', test_video2.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.expand_dims(test_video2,axis=1)
                                                else:
                                                    test_video2=np.expand_dims(test_video2,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.rollaxis(test_video2,3,1)
                                                else:
                                                    test_video2=np.rollaxis(test_video2,3,4)
                                              
                                            #*****************************************
                                            print(test_video2.shape)                      
                                            test_video2=Normalize(test_video2)                     
                                            out_kelass15 = model27.predict_classes(test_video2)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass15==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass15==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass15==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass15==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_2 = []
                                            listdata_frame_person2 = []
                                        
                                        
                                        
                                    if mlb.classes_[j]=='yellow':
                                        self.counter2_2 += 1
                                        person_2.append(next)
                                        text_count=  "ID: 8"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_2==5:        
                                            input2 = np.array(person_2)
                                            #print(input.shape)
                                            input2 = input2.astype('float32')
                                            listdata_frame_person2.append(input2)
                                            self.counter2_2=0
                                            test_video2 = np.array(listdata_frame_person2)
                                            num_of_samples2=len(test_video2)
                                            print('Number of samples5: ', num_of_samples2)
                                            print('data_shape5: ', test_video2.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.expand_dims(test_video2,axis=1)
                                                else:
                                                    test_video2=np.expand_dims(test_video2,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.rollaxis(test_video2,3,1)
                                                else:
                                                    test_video2=np.rollaxis(test_video2,3,4)
                                              
                                            #*****************************************
                                            print(test_video2.shape)                      
                                            test_video2=Normalize(test_video2)                     
                                            out_kelass16 = model28.predict_classes(test_video2)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass16==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass16==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass16==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass16==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_2 = []
                                            listdata_frame_person2 = []
                                            
                                            
                                            
                                            
                                        
                                    if mlb.classes_[j]=='black':
                                        self.counter2_2 += 1
                                        person_2.append(next)
                                        text_count=  "ID: 9"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_2==5:        
                                            input2 = np.array(person_2)
                                            #print(input.shape)
                                            input2 = input2.astype('float32')
                                            listdata_frame_person2.append(input2)
                                            self.counter2_2=0
                                            test_video2 = np.array(listdata_frame_person2)
                                            num_of_samples2=len(test_video2)
                                            print('Number of samples5: ', num_of_samples2)
                                            print('data_shape5: ', test_video2.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.expand_dims(test_video2,axis=1)
                                                else:
                                                    test_video2=np.expand_dims(test_video2,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video2=np.rollaxis(test_video2,3,1)
                                                else:
                                                    test_video2=np.rollaxis(test_video2,3,4)
                                              
                                            #*****************************************
                                            print(test_video2.shape)                      
                                            test_video2=Normalize(test_video2)                     
                                            out_kelass67 = model29.predict_classes(test_video2)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass67==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass67==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass67==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass67==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_2 = []
                                            listdata_frame_person2 = []
   
                        
                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        
                        if last_item==3: # agar akharin tedade mashine k tashkhis dadi 2 bod?
                                #count2 = 2
                                #print(self.count2)
                                #text_count=  " " + ('%.d' %count2)
                                #cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                #if num_car[1]=='2': 
                                frame3 = cv2.resize(letter,(96,96))
                                frame3 = frame2.astype("float") / 255.0
                                input_data3 = img_to_array(frame3) 
                                input_data3 = np.expand_dims(input_data3, axis=0)
                                #print(input_data2.shape)
                                    
                                # labels 
                                #print("[INFO] classifying image...")
                                proba3 = model3.predict(input_data3)[0]
                                idxs3 = np.argsort(proba3)[::-1][:2]
                            
                                # loop over 
                                for (i, j) in enumerate(idxs3):
                                    # build the label and draw the label on the image
                                    label3 = "{}: {:.2f}%".format(mlb.classes_[j], proba3[j] * 100)
                                    cv2.putText(letter, label3, (385, (i * 30) + 25), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    if mlb.classes_[j]=='blue':
                                        self.counter2_3 += 1
                                        person_3.append(next)
                                        text_count=  "ID: 1"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_3==5:
                                            input3 = np.array(person_3)
                                            input3 = input3.astype('float32')
                                            listdata_frame_person3.append(input3)
                                            self.counter2_3=0
                                            test_video3 = np.array(listdata_frame_person3)
                                            num_of_samples3=len(test_video3)
                                            print('Number of samples: ', num_of_samples3)
                                            print('data_shape: ', test_video3.shape)
                                            
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.expand_dims(test_video3,axis=1)
                                                else:
                                                    test_video3=np.expand_dims(test_video3,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.rollaxis(test_video3,3,1)
                                                else:
                                                    test_video3=np.rollaxis(test_video3,3,4)
                                                                    
                                    #*************************************************     
                                            test_video3=Normalize(test_video3)
                                            out_kelass17 = model21.predict_classes(test_video3)
        
                                    #*************************************************
        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            if out_kelass17==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                                
                                            elif out_kelass17==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown_action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass17==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass17==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            person_3 = []
                                            listdata_frame_person3 = []
                                            
                                    if mlb.classes_[j]=='brown':
                                        self.counter2_3 += 1
                                        person_3.append(next)
                                        text_count=  "ID: 2"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_3==5:      
                                            input3 = np.array(person_3)
                                            #print(input.shape)
                                            input3 = input3.astype('float32')
                                            listdata_frame_person3.append(input3)
                                            self.counter2_3=0
                                            test_video3 = np.array(listdata_frame_person3)
                                            num_of_samples3=len(test_video3)
                                            print('Number of samples2: ', num_of_samples3)
                                            print('data_shape2: ', test_video3.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.expand_dims(test_video3,axis=1)
                                                else:
                                                    test_video3=np.expand_dims(test_video3,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.rollaxis(test_video3,3,1)
                                                else:
                                                    test_video3=np.rollaxis(test_video3,3,4)
                                                    
                                            #****************************************
                                            print(test_video3.shape)                    
                                            test_video3=Normalize(test_video3)                     
                                            out_kelass18 = model22.predict_classes(test_video3)     
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************************************
                                            
                                            if out_kelass18==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)   
                                                
                                            elif out_kelass18==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass18==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass18==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_3 = []
                                            listdata_frame_person3 = []
                                            
                                    if mlb.classes_[j]=='green':
                                        self.counter2_3 += 1
                                        person_3.append(next)
                                        text_count=  "ID: 3"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_3==5:         
                                            input3 = np.array(person_3)
                                            #print(input.shape)
                                            input3 = input3.astype('float32')
                                            listdata_frame_person3.append(input3)
                                            self.counter2_3=0
                                            test_video3 = np.array(listdata_frame_person3)
                                            num_of_samples3=len(test_video3)
                                            print('Number of samples3: ', num_of_samples3)
                                            print('data_shape3: ', test_video3.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.expand_dims(test_video3,axis=1)
                                                else:
                                                    test_video3=np.expand_dims(test_video3,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.rollaxis(test_video3,3,1)
                                                else:
                                                    test_video3=np.rollaxis(test_video3,3,4)
                                                    
                                            #*****************************************
                                            print(test_video3.shape)                       
                                            test_video3=Normalize(test_video3)                     
                                            out_kelass19= model23.predict_classes(test_video3)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************** **********************                                   
                                            
                                            if out_kelass19==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass19==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass19==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass19==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_3 = []
                                            listdata_frame_person3 = []  
        
        
                                    if mlb.classes_[j]=='orange':
                                        self.counter2_3 += 1
                                        person_3.append(next)
                                        text_count=  "ID: 4"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_3==5:       
                                            input3 = np.array(person_3)
                                            #print(input.shape)
                                            input3 = input3.astype('float32')
                                            listdata_frame_person3.append(input3)
                                            self.counter2_3=0
                                            test_video3 = np.array(listdata_frame_person3)
                                            num_of_samples3=len(test_video3)
                                            print('Number of samples4: ', num_of_samples3)
                                            print('data_shape4: ', test_video3.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.expand_dims(test_video3,axis=1)
                                                else:
                                                    test_video3=np.expand_dims(test_video3,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.rollaxis(test_video3,3,1)
                                                else:
                                                    test_video3=np.rollaxis(test_video3,3,4)
                                                    
                                            #*****************************************
                                            print(test_video3.shape)                      
                                            test_video3=Normalize(test_video3)                
                                            out_kelass20 = model24.predict_classes(test_video3)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass20==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass20==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass20==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass20==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_3 = []
                                            listdata_frame_person3 = []
                                        
                                        
                                    if mlb.classes_[j]=='purple':
                                        self.counter2_3 += 1
                                        person_3.append(next)
                                        text_count=  "ID: 5"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_3==5:        
                                            input3 = np.array(person_3)
                                            #print(input.shape)
                                            input3 = input3.astype('float32')
                                            listdata_frame_person3.append(input3)
                                            self.counter2_3=0
                                            test_video3 = np.array(listdata_frame_person3)
                                            num_of_samples3=len(test_video3)
                                            print('Number of samples5: ', num_of_samples3)
                                            print('data_shape5: ', test_video3.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.expand_dims(test_video3,axis=1)
                                                else:
                                                    test_video3=np.expand_dims(test_video3,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.rollaxis(test_video3,3,1)
                                                else:
                                                    test_video3=np.rollaxis(test_video3,3,4)
                                              
                                            #*****************************************
                                            print(test_video3.shape)                      
                                            test_video3=Normalize(test_video3)                     
                                            out_kelass21 = model25.predict_classes(test_video3)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass21==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass21==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass21==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass21==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_3 = []
                                            listdata_frame_person3 = []                 
        
            
                                    if mlb.classes_[j]=='red':
                                        self.counter2_3 += 1
                                        person_3.append(next)
                                        #self.counter2 += 1
                                        #print(self.counter2)
                                        text_count=  "ID: 6"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_3==5:        
                                            input3 = np.array(person_3)
                                            #print(input.shape)
                                            input3 = input3.astype('float32')
                                            listdata_frame_person3.append(input3)
                                            self.counter2_3=0
                                            test_video3 = np.array(listdata_frame_person3)
                                            num_of_samples3=len(test_video3)
                                            print('Number of samples5: ', num_of_samples3)
                                            print('data_shape5: ', test_video3.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.expand_dims(test_video3,axis=1)
                                                else:
                                                    test_video3=np.expand_dims(test_video3,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.rollaxis(test_video3,3,1)
                                                else:
                                                    test_video3=np.rollaxis(test_video3,3,4)
                                              
                                            #*****************************************
                                            print(test_video3.shape)                      
                                            test_video3=Normalize(test_video3)                     
                                            out_kelass22 = model26.predict_classes(test_video3)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass22==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass22==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass22==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass22==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_3 = []
                                            listdata_frame_person3 = []
        
        
                                    if mlb.classes_[j]=='white':
                                        self.counter2_3 += 1
                                        person_3.append(next)
                                        text_count=  "ID: 7"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_3==5:        
                                            input3 = np.array(person_3)
                                            #print(input.shape)
                                            input3 = input3.astype('float32')
                                            listdata_frame_person3.append(input3)
                                            self.counter2_3=0
                                            test_video3 = np.array(listdata_frame_person3)
                                            num_of_samples3=len(test_video3)
                                            print('Number of samples5: ', num_of_samples3)
                                            print('data_shape5: ', test_video3.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.expand_dims(test_video3,axis=1)
                                                else:
                                                    test_video3=np.expand_dims(test_video3,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.rollaxis(test_video3,3,1)
                                                else:
                                                    test_video3=np.rollaxis(test_video3,3,4)
                                              
                                            #*****************************************
                                            print(test_video3.shape)                      
                                            test_video3=Normalize(test_video3)                     
                                            out_kelass23 = model27.predict_classes(test_video3)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass23==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass23==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass23==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass23==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_3 = []
                                            listdata_frame_person3 = []
                                        
                                        
                                        
                                    if mlb.classes_[j]=='yellow':
                                        self.counter2_3 += 1
                                        person_3.append(next)
                                        text_count=  "ID: 8"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_3==5:        
                                            input3 = np.array(person_3)
                                            #print(input.shape)
                                            input3 = input3.astype('float32')
                                            listdata_frame_person3.append(input3)
                                            self.counter2_3=0
                                            test_video3 = np.array(listdata_frame_person3)
                                            num_of_samples3=len(test_video3)
                                            print('Number of samples5: ', num_of_samples3)
                                            print('data_shape5: ', test_video3.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.expand_dims(test_video3,axis=1)
                                                else:
                                                    test_video3=np.expand_dims(test_video3,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.rollaxis(test_video3,3,1)
                                                else:
                                                    test_video3=np.rollaxis(test_video3,3,4)
                                              
                                            #*****************************************
                                            print(test_video3.shape)                      
                                            test_video3=Normalize(test_video3)                     
                                            out_kelass24 = model28.predict_classes(test_video3)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass24==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass24==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass24==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass24==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_3 = []
                                            listdata_frame_person3 = []
                                            
                                            
                                            
                                    if mlb.classes_[j]=='black':
                                        self.counter2_3 += 1
                                        person_3.append(next)
                                        text_count=  "ID: 9"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_3==5:        
                                            input3 = np.array(person_3)
                                            #print(input.shape)
                                            input3 = input3.astype('float32')
                                            listdata_frame_person3.append(input3)
                                            self.counter2_3=0
                                            test_video3 = np.array(listdata_frame_person3)
                                            num_of_samples3=len(test_video3)
                                            print('Number of samples5: ', num_of_samples3)
                                            print('data_shape5: ', test_video3.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.expand_dims(test_video3,axis=1)
                                                else:
                                                    test_video3=np.expand_dims(test_video3,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video3=np.rollaxis(test_video3,3,1)
                                                else:
                                                    test_video3=np.rollaxis(test_video3,3,4)
                                              
                                            #*****************************************
                                            print(test_video3.shape)                      
                                            test_video3=Normalize(test_video3)                     
                                            out_kelass68 = model29.predict_classes(test_video3)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass68==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass68==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass68==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass68==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_3 = []
                                            listdata_frame_person3 = []
                              
                              
                        
                        
                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if last_item==4: # agar akharin tedade mashine k tashkhis dadi 2 bod?
                                #count2 = 2
                                #print(self.count2)
                                #text_count=  " " + ('%.d' %count2)
                                #cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                #if num_car[1]=='2': 
                                frame4 = cv2.resize(letter,(96,96))
                                frame4 = frame4.astype("float") / 255.0
                                input_data4 = img_to_array(frame4) 
                                input_data4 = np.expand_dims(input_data4, axis=0)
                                #print(input_data2.shape)
                                    
                                # labels 
                                #print("[INFO] classifying image...")
                                proba4 = model4.predict(input_data4)[0]
                                idxs4 = np.argsort(proba4)[::-1][:2]
                            
                                # loop over 
                                for (i, j) in enumerate(idxs4):
                                    # build the label and draw the label on the image
                                    label4 = "{}: {:.2f}%".format(mlb.classes_[j], proba4[j] * 100)
                                    cv2.putText(letter, label4, (385, (i * 30) + 25), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    if mlb.classes_[j]=='blue':
                                        self.counter2_4 += 1
                                        person_4.append(next)
                                        text_count=  "ID: 1"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_4==5:
                                            input4 = np.array(person_4)
                                            input4 = input4.astype('float32')
                                            listdata_frame_person4.append(input4)
                                            self.counter2_4=0
                                            test_video4 = np.array(listdata_frame_person4)
                                            num_of_samples4=len(test_video4)
                                            print('Number of samples: ', num_of_samples4)
                                            print('data_shape: ', test_video4.shape)
                                            
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.expand_dims(test_video4,axis=1)
                                                else:
                                                    test_video4=np.expand_dims(test_video4,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.rollaxis(test_video4,3,1)
                                                else:
                                                    test_video4=np.rollaxis(test_video4,3,4)
                                                                    
                                    #*************************************************     
                                            test_video4=Normalize(test_video4)
                                            out_kelass25 = model21.predict_classes(test_video4)
        
                                    #*************************************************
        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            if out_kelass25==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                                
                                            elif out_kelass25==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown_action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass25==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass25==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            person_4= []
                                            listdata_frame_person4 = []
                                            
                                    if mlb.classes_[j]=='brown':
                                        self.counter2_4 += 1
                                        person_4.append(next)
                                        text_count=  "ID: 2"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_4==5:      
                                            input4 = np.array(person_4)
                                            #print(input.shape)
                                            input4 = input4.astype('float32')
                                            listdata_frame_person4.append(input4)
                                            self.counter2_4=0
                                            test_video4 = np.array(listdata_frame_person4)
                                            num_of_samples4=len(test_video4)
                                            print('Number of samples2: ', num_of_samples4)
                                            print('data_shape2: ', test_video4.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.expand_dims(test_video4,axis=1)
                                                else:
                                                    test_video4=np.expand_dims(test_video4,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.rollaxis(test_video4,3,1)
                                                else:
                                                    test_video4=np.rollaxis(test_video4,3,4)
                                                    
                                            #****************************************
                                            print(test_video4.shape)                    
                                            test_video4=Normalize(test_video4)                     
                                            out_kelass26 = model22.predict_classes(test_video4)     
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************************************
                                            
                                            if out_kelass26==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)   
                                                
                                            elif out_kelass26==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass26==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass26==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_4 = []
                                            listdata_frame_person4 = []
                                            
                                    if mlb.classes_[j]=='green':
                                        self.counter2_4 += 1
                                        person_4.append(next)
                                        text_count=  "ID: 3"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_4==5:         
                                            input4 = np.array(person_4)
                                            #print(input.shape)
                                            input4 = input4.astype('float32')
                                            listdata_frame_person4.append(input4)
                                            self.counter2_4=0
                                            test_video4 = np.array(listdata_frame_person4)
                                            num_of_samples4=len(test_video4)
                                            print('Number of samples3: ', num_of_samples4)
                                            print('data_shape3: ', test_video4.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.expand_dims(test_video4,axis=1)
                                                else:
                                                    test_video4=np.expand_dims(test_video4,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.rollaxis(test_video4,3,1)
                                                else:
                                                    test_video4=np.rollaxis(test_video4,3,4)
                                                    
                                            #*****************************************
                                            print(test_video4.shape)                       
                                            test_video4=Normalize(test_video4)                     
                                            out_kelass27= model23.predict_classes(test_video4)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************** **********************                                   
                                            
                                            if out_kelass27==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass27==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass27==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass27==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_4 = []
                                            listdata_frame_person4 = []  
        
        
                                    if mlb.classes_[j]=='orange':
                                        self.counter2_4 += 1
                                        person_4.append(next)
                                        text_count=  "ID: 4"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_4==5:       
                                            input4 = np.array(person_4)
                                            #print(input.shape)
                                            input4 = input4.astype('float32')
                                            listdata_frame_person4.append(input4)
                                            self.counter2_4=0
                                            test_video4 = np.array(listdata_frame_person4)
                                            num_of_samples4=len(test_video4)
                                            print('Number of samples4: ', num_of_samples4)
                                            print('data_shape4: ', test_video4.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.expand_dims(test_video4,axis=1)
                                                else:
                                                    test_video4=np.expand_dims(test_video4,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.rollaxis(test_video4,3,1)
                                                else:
                                                    test_video4=np.rollaxis(test_video4,3,4)
                                                    
                                            #*****************************************
                                            print(test_video4.shape)                      
                                            test_video4=Normalize(test_video4)                
                                            out_kelass28 = model24.predict_classes(test_video4)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass28==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass28==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass28==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass28==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_4 = []
                                            listdata_frame_person4 = []
                                        
                                        
                                    if mlb.classes_[j]=='purple':
                                        self.counter2_4 += 1
                                        person_4.append(next)
                                        text_count=  "ID: 5"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_4==5:        
                                            input4 = np.array(person_4)
                                            #print(input.shape)
                                            input4 = input4.astype('float32')
                                            listdata_frame_person4.append(input4)
                                            self.counter2_4=0
                                            test_video4 = np.array(listdata_frame_person4)
                                            num_of_samples4=len(test_video4)
                                            print('Number of samples5: ', num_of_samples4)
                                            print('data_shape5: ', test_video4.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.expand_dims(test_video4,axis=1)
                                                else:
                                                    test_video4=np.expand_dims(test_video4,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.rollaxis(test_video4,3,1)
                                                else:
                                                    test_video4=np.rollaxis(test_video4,3,4)
                                              
                                            #*****************************************
                                            print(test_video4.shape)                      
                                            test_video4=Normalize(test_video4)                     
                                            out_kelass29 = model25.predict_classes(test_video4)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass29==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass29==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass29==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass29==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_4 = []
                                            listdata_frame_person4 = []                 
        
            
                                    if mlb.classes_[j]=='red':
                                        self.counter2_4 += 1
                                        person_4.append(next)
                                        #self.counter2 += 1
                                        #print(self.counter2)
                                        text_count=  "ID: 6"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_4==5:        
                                            input4 = np.array(person_4)
                                            #print(input.shape)
                                            input4 = input4.astype('float32')
                                            listdata_frame_person4.append(input4)
                                            self.counter2_4=0
                                            test_video4 = np.array(listdata_frame_person4)
                                            num_of_samples4=len(test_video4)
                                            print('Number of samples5: ', num_of_samples4)
                                            print('data_shape5: ', test_video4.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.expand_dims(test_video4,axis=1)
                                                else:
                                                    test_video4=np.expand_dims(test_video4,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.rollaxis(test_video4,3,1)
                                                else:
                                                    test_video4=np.rollaxis(test_video4,3,4)
                                              
                                            #*****************************************
                                            print(test_video4.shape)                      
                                            test_video4=Normalize(test_video4)                     
                                            out_kelass30 = model26.predict_classes(test_video4)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass30==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass30==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass30==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass30==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_4 = []
                                            listdata_frame_person4 = []
        
        
                                    if mlb.classes_[j]=='white':
                                        self.counter2_4 += 1
                                        person_4.append(next)
                                        text_count=  "ID: 7"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_4==5:        
                                            input4 = np.array(person_4)
                                            #print(input.shape)
                                            input4 = input4.astype('float32')
                                            listdata_frame_person4.append(input4)
                                            self.counter2_4=0
                                            test_video4 = np.array(listdata_frame_person4)
                                            num_of_samples4=len(test_video4)
                                            print('Number of samples5: ', num_of_samples4)
                                            print('data_shape5: ', test_video4.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.expand_dims(test_video4,axis=1)
                                                else:
                                                    test_video4=np.expand_dims(test_video4,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.rollaxis(test_video4,3,1)
                                                else:
                                                    test_video4=np.rollaxis(test_video4,3,4)
                                              
                                            #*****************************************
                                            print(test_video4.shape)                      
                                            test_video4=Normalize(test_video4)                     
                                            out_kelass31 = model27.predict_classes(test_video4)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass31==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass31==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass31==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass31==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_4 = []
                                            listdata_frame_person4 = []
                                        
                                        
                                        
                                    if mlb.classes_[j]=='yellow':
                                        self.counter2_4 += 1
                                        person_4.append(next)
                                        text_count=  "ID: 8"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_4==5:        
                                            input4 = np.array(person_4)
                                            #print(input.shape)
                                            input4 = input4.astype('float32')
                                            listdata_frame_person4.append(input4)
                                            self.counter2_4=0
                                            test_video4 = np.array(listdata_frame_person4)
                                            num_of_samples4=len(test_video4)
                                            print('Number of samples5: ', num_of_samples4)
                                            print('data_shape5: ', test_video4.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.expand_dims(test_video4,axis=1)
                                                else:
                                                    test_video4=np.expand_dims(test_video4,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.rollaxis(test_video4,3,1)
                                                else:
                                                    test_video4=np.rollaxis(test_video4,3,4)
                                              
                                            #*****************************************
                                            print(test_video4.shape)                      
                                            test_video4=Normalize(test_video4)                     
                                            out_kelass32 = model28.predict_classes(test_video4)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass32==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass32==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass32==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass32==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_4 = []
                                            listdata_frame_person4 = []
                                            
                                            
                                            
                                            
                                    if mlb.classes_[j]=='black':
                                        self.counter2_4 += 1
                                        person_4.append(next)
                                        text_count=  "ID: 9"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_4==5:        
                                            input4 = np.array(person_4)
                                            #print(input.shape)
                                            input4 = input4.astype('float32')
                                            listdata_frame_person4.append(input4)
                                            self.counter2_4=0
                                            test_video4 = np.array(listdata_frame_person4)
                                            num_of_samples4=len(test_video4)
                                            print('Number of samples5: ', num_of_samples4)
                                            print('data_shape5: ', test_video4.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.expand_dims(test_video4,axis=1)
                                                else:
                                                    test_video4=np.expand_dims(test_video4,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video4=np.rollaxis(test_video4,3,1)
                                                else:
                                                    test_video4=np.rollaxis(test_video4,3,4)
                                              
                                            #*****************************************
                                            print(test_video4.shape)                      
                                            test_video4=Normalize(test_video4)                     
                                            out_kelass69 = model29.predict_classes(test_video4)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass69==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass69==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass69==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass69==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_4 = []
                                            listdata_frame_person4 = []
                                        
                                        
                                        
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                        
                        if last_item==5: # agar akharin tedade mashine k tashkhis dadi 2 bod?
                                #count2 = 2
                                #print(self.count2)
                                #text_count=  " " + ('%.d' %count2)
                                #cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                #if num_car[1]=='2': 
                                frame5 = cv2.resize(letter,(96,96))
                                frame5 = frame2.astype("float") / 255.0
                                input_data5 = img_to_array(frame5) 
                                input_data5 = np.expand_dims(input_data5, axis=0)
                                #print(input_data2.shape)
                                    
                                # labels 
                                #print("[INFO] classifying image...")
                                proba5 = model5.predict(input_data5)[0]
                                idxs5 = np.argsort(proba5)[::-1][:2]
                            
                                # loop over 
                                for (i, j) in enumerate(idxs5):
                                    # build the label and draw the label on the image
                                    label5 = "{}: {:.2f}%".format(mlb.classes_[j], proba5[j] * 100)
                                    cv2.putText(letter, label5, (385, (i * 30) + 25), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    
                                    if mlb.classes_[j]=='blue':
                                        self.counter2_5 += 1
                                        person_5.append(next)
                                        text_count=  "ID: 1"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_5==5:
                                            input5 = np.array(person_5)
                                            input5 = input5.astype('float32')
                                            listdata_frame_person5.append(input5)
                                            self.counter2_5=0
                                            test_video5 = np.array(listdata_frame_person5)
                                            num_of_samples5=len(test_video5)
                                            print('Number of samples: ', num_of_samples5)
                                            print('data_shape: ', test_video5.shape)
                                            
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.expand_dims(test_video5,axis=1)
                                                else:
                                                    test_video5=np.expand_dims(test_video5,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.rollaxis(test_video5,3,1)
                                                else:
                                                    test_video5=np.rollaxis(test_video5,3,4)
                                                                    
                                    #*************************************************     
                                            test_video5=Normalize(test_video5)
                                            out_kelass33 = model21.predict_classes(test_video5)
        
                                    #*************************************************
        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            if out_kelass33==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                                
                                            elif out_kelass33==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown_action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass33==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass33==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            person_5= []
                                            listdata_frame_person5 = []
                                            
                                    if mlb.classes_[j]=='brown':
                                        self.counter2_5 += 1
                                        person_5.append(next)
                                        text_count=  "ID: 2"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_5==5:      
                                            input5 = np.array(person_5)
                                            #print(input.shape)
                                            input5 = input5.astype('float32')
                                            listdata_frame_person5.append(input5)
                                            self.counter2_5=0
                                            test_video5 = np.array(listdata_frame_person5)
                                            num_of_samples5=len(test_video5)
                                            print('Number of samples2: ', num_of_samples5)
                                            print('data_shape2: ', test_video5.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.expand_dims(test_video5,axis=1)
                                                else:
                                                    test_video5=np.expand_dims(test_video5,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.rollaxis(test_video5,3,1)
                                                else:
                                                    test_video5=np.rollaxis(test_video5,3,4)
                                                    
                                            #****************************************
                                            print(test_video5.shape)                    
                                            test_video5=Normalize(test_video5)                     
                                            out_kelass34 = model22.predict_classes(test_video5)     
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************************************
                                            
                                            if out_kelass34==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)   
                                                
                                            elif out_kelass34==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass34==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass34==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_5 = []
                                            listdata_frame_person5 = []
                                            
                                    if mlb.classes_[j]=='green':
                                        self.counter2_5 += 1
                                        person_5.append(next)
                                        text_count=  "ID: 3"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_5==5:         
                                            input5 = np.array(person_5)
                                            #print(input.shape)
                                            input5 = input5.astype('float32')
                                            listdata_frame_person5.append(input5)
                                            self.counter2_5=0
                                            test_video5 = np.array(listdata_frame_person5)
                                            num_of_samples5=len(test_video5)
                                            print('Number of samples3: ', num_of_samples5)
                                            print('data_shape3: ', test_video5.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.expand_dims(test_video5,axis=1)
                                                else:
                                                    test_video5=np.expand_dims(test_video5,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.rollaxis(test_video5,3,1)
                                                else:
                                                    test_video5=np.rollaxis(test_video5,3,4)
                                                    
                                            #*****************************************
                                            print(test_video5.shape)                       
                                            test_video5=Normalize(test_video5)                     
                                            out_kelass35= model23.predict_classes(test_video5)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************** **********************                                   
                                            
                                            if out_kelass35==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass35==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass35==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass35==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_5 = []
                                            listdata_frame_person5 = []  
        
        
                                    if mlb.classes_[j]=='orange':
                                        self.counter2_5 += 1
                                        person_5.append(next)
                                        text_count=  "ID: 4"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_5==5:       
                                            input5 = np.array(person_5)
                                            #print(input.shape)
                                            input5 = input5.astype('float32')
                                            listdata_frame_person5.append(input5)
                                            self.counter2_5=0
                                            test_video5 = np.array(listdata_frame_person5)
                                            num_of_samples5=len(test_video5)
                                            print('Number of samples4: ', num_of_samples5)
                                            print('data_shape4: ', test_video5.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.expand_dims(test_video5,axis=1)
                                                else:
                                                    test_video5=np.expand_dims(test_video5,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.rollaxis(test_video5,3,1)
                                                else:
                                                    test_video5=np.rollaxis(test_video5,3,4)
                                                    
                                            #*****************************************
                                            print(test_video5.shape)                      
                                            test_video5=Normalize(test_video5)                
                                            out_kelass36 = model24.predict_classes(test_video5)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass36==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass36==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass36==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass36==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_5 = []
                                            listdata_frame_person5 = []
                                        
                                        
                                    if mlb.classes_[j]=='purple':
                                        self.counter2_5 += 1
                                        person_5.append(next)
                                        text_count=  "ID: 5"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_5==5:        
                                            input5 = np.array(person_5)
                                            #print(input.shape)
                                            input5 = input5.astype('float32')
                                            listdata_frame_person5.append(input5)
                                            self.counter2_5=0
                                            test_video5 = np.array(listdata_frame_person5)
                                            num_of_samples5=len(test_video4)
                                            print('Number of samples5: ', num_of_samples5)
                                            print('data_shape5: ', test_video5.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.expand_dims(test_video5,axis=1)
                                                else:
                                                    test_video5=np.expand_dims(test_video5,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.rollaxis(test_video5,3,1)
                                                else:
                                                    test_video5=np.rollaxis(test_video5,3,4)
                                              
                                            #*****************************************
                                            print(test_video5.shape)                      
                                            test_video5=Normalize(test_video5)                     
                                            out_kelass37 = model25.predict_classes(test_video5)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass37==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass37==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass37==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass37==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_5 = []
                                            listdata_frame_person5 = []                 
        
            
                                    if mlb.classes_[j]=='red':
                                        self.counter2_5 += 1
                                        person_5.append(next)
                                        #self.counter2 += 1
                                        #print(self.counter2)
                                        text_count=  "ID: 6"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_5==5:        
                                            input5 = np.array(person_5)
                                            #print(input.shape)
                                            input5 = input5.astype('float32')
                                            listdata_frame_person5.append(input5)
                                            self.counter2_5=0
                                            test_video5 = np.array(listdata_frame_person5)
                                            num_of_samples5=len(test_video5)
                                            print('Number of samples5: ', num_of_samples5)
                                            print('data_shape5: ', test_video5.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.expand_dims(test_video5,axis=1)
                                                else:
                                                    test_video5=np.expand_dims(test_video5,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.rollaxis(test_video5,3,1)
                                                else:
                                                    test_video5=np.rollaxis(test_video5,3,4)
                                              
                                            #*****************************************
                                            print(test_video5.shape)                      
                                            test_video5=Normalize(test_video5)                     
                                            out_kelass38 = model26.predict_classes(test_video5)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass38==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass38==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass38==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass38==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_5 = []
                                            listdata_frame_person5 = []
        
        
                                    if mlb.classes_[j]=='white':
                                        self.counter2_5 += 1
                                        person_5.append(next)
                                        text_count=  "ID: 7"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_5==5:        
                                            input5 = np.array(person_5)
                                            #print(input.shape)
                                            input5 = input5.astype('float32')
                                            listdata_frame_person5.append(input5)
                                            self.counter2_5=0
                                            test_video5 = np.array(listdata_frame_person5)
                                            num_of_samples5=len(test_video5)
                                            print('Number of samples5: ', num_of_samples5)
                                            print('data_shape5: ', test_video5.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.expand_dims(test_video5,axis=1)
                                                else:
                                                    test_video5=np.expand_dims(test_video5,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.rollaxis(test_video5,3,1)
                                                else:
                                                    test_video5=np.rollaxis(test_video5,3,4)
                                              
                                            #*****************************************
                                            print(test_video5.shape)                      
                                            test_video5=Normalize(test_video5)                     
                                            out_kelass39 = model27.predict_classes(test_video5)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass39==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass39==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass39==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass39==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_5 = []
                                            listdata_frame_person5 = []
                                        
                                        
                                        
                                    if mlb.classes_[j]=='yellow':
                                        self.counter2_5 += 1
                                        person_5.append(next)
                                        text_count=  "ID: 8"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_5==5:        
                                            input5 = np.array(person_5)
                                            #print(input.shape)
                                            input5 = input5.astype('float32')
                                            listdata_frame_person5.append(input5)
                                            self.counter2_5=0
                                            test_video5 = np.array(listdata_frame_person5)
                                            num_of_samples5=len(test_video5)
                                            print('Number of samples5: ', num_of_samples5)
                                            print('data_shape5: ', test_video5.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.expand_dims(test_video5,axis=1)
                                                else:
                                                    test_video5=np.expand_dims(test_video5,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.rollaxis(test_video5,3,1)
                                                else:
                                                    test_video5=np.rollaxis(test_video5,3,4)
                                              
                                            #*****************************************
                                            print(test_video5.shape)                      
                                            test_video5=Normalize(test_video5)                     
                                            out_kelass40 = model28.predict_classes(test_video5)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass40==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass40==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass40==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass40==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_5 = []
                                            listdata_frame_person5 = []
                                            
                                            
                                            
                                            
                                    if mlb.classes_[j]=='black':
                                        self.counter2_5 += 1
                                        person_5.append(next)
                                        text_count=  "ID: 9"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_5==5:        
                                            input5 = np.array(person_5)
                                            #print(input.shape)
                                            input5 = input5.astype('float32')
                                            listdata_frame_person5.append(input5)
                                            self.counter2_5=0
                                            test_video5 = np.array(listdata_frame_person5)
                                            num_of_samples5=len(test_video5)
                                            print('Number of samples5: ', num_of_samples5)
                                            print('data_shape5: ', test_video5.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.expand_dims(test_video5,axis=1)
                                                else:
                                                    test_video5=np.expand_dims(test_video5,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video5=np.rollaxis(test_video5,3,1)
                                                else:
                                                    test_video5=np.rollaxis(test_video5,3,4)
                                              
                                            #*****************************************
                                            print(test_video5.shape)                      
                                            test_video5=Normalize(test_video5)                     
                                            out_kelass70 = model29.predict_classes(test_video5)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass70==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass70==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass70==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass70==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_5 = []
                                            listdata_frame_person5 = []
                                            
                                            
                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                        
                        if last_item==6: # agar akharin tedade mashine k tashkhis dadi 2 bod?
                                #count2 = 2
                                #print(self.count2)
                                #text_count=  " " + ('%.d' %count2)
                                #cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                #if num_car[1]=='2': 
                                frame6 = cv2.resize(letter,(96,96))
                                frame6 = frame6.astype("float") / 255.0
                                input_data6 = img_to_array(frame6) 
                                input_data6 = np.expand_dims(input_data6, axis=0)
                                #print(input_data2.shape)
                                    
                                # labels 
                                #print("[INFO] classifying image...")
                                proba6 = model6.predict(input_data6)[0]
                                idxs6 = np.argsort(proba6)[::-1][:2]
                            
                                # loop over 
                                for (i, j) in enumerate(idxs6):
                                    # build the label and draw the label on the image
                                    label6 = "{}: {:.2f}%".format(mlb.classes_[j], proba6[j] * 100)
                                    cv2.putText(letter, label6, (385, (i * 30) + 25), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    if mlb.classes_[j]=='blue':
                                        self.counter2_6 += 1
                                        person_6.append(next)
                                        text_count=  "ID: 1"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_6==5:
                                            input6 = np.array(person_6)
                                            input6 = input6.astype('float32')
                                            listdata_frame_person6.append(input6)
                                            self.counter2_6=0
                                            test_video6 = np.array(listdata_frame_person6)
                                            num_of_samples6=len(test_video6)
                                            print('Number of samples: ', num_of_samples6)
                                            print('data_shape: ', test_video6.shape)
                                            
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.expand_dims(test_video6,axis=1)
                                                else:
                                                    test_video6=np.expand_dims(test_video6,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.rollaxis(test_video6,3,1)
                                                else:
                                                    test_video6=np.rollaxis(test_video6,3,4)
                                                                    
                                    #*************************************************     
                                            test_video6=Normalize(test_video6)
                                            out_kelass41 = model21.predict_classes(test_video6)
        
                                    #*************************************************
        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            if out_kelass41==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                                
                                            elif out_kelass41==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown_action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass41==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass41==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            person_6= []
                                            listdata_frame_person6 = []
                                            
                                    if mlb.classes_[j]=='brown':
                                        self.counter2_6 += 1
                                        person_6.append(next)
                                        text_count=  "ID: 2"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_6==5:      
                                            input6 = np.array(person_6)
                                            #print(input.shape)
                                            input6 = input6.astype('float32')
                                            listdata_frame_person6.append(input6)
                                            self.counter2_6=0
                                            test_video6 = np.array(listdata_frame_person6)
                                            num_of_samples6=len(test_video6)
                                            print('Number of samples2: ', num_of_samples6)
                                            print('data_shape2: ', test_video6.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.expand_dims(test_video6,axis=1)
                                                else:
                                                    test_video6=np.expand_dims(test_video6,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.rollaxis(test_video6,3,1)
                                                else:
                                                    test_video6=np.rollaxis(test_video6,3,4)
                                                    
                                            #****************************************
                                            print(test_video6.shape)                    
                                            test_video6=Normalize(test_video6)                     
                                            out_kelass42 = model22.predict_classes(test_video6)     
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************************************
                                            
                                            if out_kelass42==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)   
                                                
                                            elif out_kelass42==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass42==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass42==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_6 = []
                                            listdata_frame_person6 = []
                                            
                                    if mlb.classes_[j]=='green':
                                        self.counter2_6 += 1
                                        person_6.append(next)
                                        text_count=  "ID: 3"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_6==5:         
                                            input6 = np.array(person_6)
                                            #print(input.shape)
                                            input6 = input6.astype('float32')
                                            listdata_frame_person6.append(input6)
                                            self.counter2_6=0
                                            test_video6 = np.array(listdata_frame_person6)
                                            num_of_samples6=len(test_video6)
                                            print('Number of samples3: ', num_of_samples6)
                                            print('data_shape3: ', test_video6.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.expand_dims(test_video6,axis=1)
                                                else:
                                                    test_video6=np.expand_dims(test_video6,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.rollaxis(test_video6,3,1)
                                                else:
                                                    test_video6=np.rollaxis(test_video6,3,4)
                                                    
                                            #*****************************************
                                            print(test_video6.shape)                       
                                            test_video6=Normalize(test_video6)                     
                                            out_kelass43= model23.predict_classes(test_video6)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************** **********************                                   
                                            
                                            if out_kelass43==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass43==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass43==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass43==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_6 = []
                                            listdata_frame_person6 = []  
        
        
                                    if mlb.classes_[j]=='orange':
                                        self.counter2_6 += 1
                                        person_6.append(next)
                                        text_count=  "ID: 4"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_6==5:       
                                            input6 = np.array(person_6)
                                            #print(input.shape)
                                            input6 = input6.astype('float32')
                                            listdata_frame_person6.append(input6)
                                            self.counter2_6=0
                                            test_video6 = np.array(listdata_frame_person6)
                                            num_of_samples6=len(test_video6)
                                            print('Number of samples4: ', num_of_samples6)
                                            print('data_shape4: ', test_video6.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.expand_dims(test_video6,axis=1)
                                                else:
                                                    test_video6=np.expand_dims(test_video6,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.rollaxis(test_video6,3,1)
                                                else:
                                                    test_video6=np.rollaxis(test_video6,3,4)
                                                    
                                            #*****************************************
                                            print(test_video6.shape)                      
                                            test_video6=Normalize(test_video6)                
                                            out_kelass44 = model24.predict_classes(test_video6)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass44==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass44==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass44==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass44==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_6 = []
                                            listdata_frame_person6 = []
                                        
                                        
                                    if mlb.classes_[j]=='purple':
                                        self.counter2_6 += 1
                                        person_6.append(next)
                                        text_count=  "ID: 5"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_6==5:        
                                            input6 = np.array(person_6)
                                            #print(input.shape)
                                            input6 = input6.astype('float32')
                                            listdata_frame_person6.append(input6)
                                            self.counter2_6=0
                                            test_video6 = np.array(listdata_frame_person6)
                                            num_of_samples6=len(test_video6)
                                            print('Number of samples5: ', num_of_samples6)
                                            print('data_shape5: ', test_video6.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.expand_dims(test_video6,axis=1)
                                                else:
                                                    test_video6=np.expand_dims(test_video6,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.rollaxis(test_video6,3,1)
                                                else:
                                                    test_video6=np.rollaxis(test_video6,3,4)
                                              
                                            #*****************************************
                                            print(test_video6.shape)                      
                                            test_video6=Normalize(test_video6)                     
                                            out_kelass45 = model25.predict_classes(test_video6)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass45==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass45==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass45==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass45==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_6 = []
                                            listdata_frame_person6 = []                 
        
            
                                    if mlb.classes_[j]=='red':
                                        self.counter2_6 += 1
                                        person_6.append(next)
                                        #self.counter2 += 1
                                        #print(self.counter2)
                                        text_count=  "ID: 6"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_6==5:        
                                            input6 = np.array(person_6)
                                            #print(input.shape)
                                            input6 = input6.astype('float32')
                                            listdata_frame_person6.append(input6)
                                            self.counter2_6=0
                                            test_video6 = np.array(listdata_frame_person6)
                                            num_of_samples6=len(test_video6)
                                            print('Number of samples5: ', num_of_samples6)
                                            print('data_shape5: ', test_video6.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.expand_dims(test_video6,axis=1)
                                                else:
                                                    test_video6=np.expand_dims(test_video6,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.rollaxis(test_video6,3,1)
                                                else:
                                                    test_video6=np.rollaxis(test_video6,3,4)
                                              
                                            #*****************************************
                                            print(test_video6.shape)                      
                                            test_video6=Normalize(test_video6)                     
                                            out_kelass46 = model26.predict_classes(test_video6)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass46==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass46==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass46==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass46==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_6 = []
                                            listdata_frame_person6 = []
        
        
                                    if mlb.classes_[j]=='white':
                                        self.counter2_6 += 1
                                        person_6.append(next)
                                        text_count=  "ID: 7"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_6==5:        
                                            input6 = np.array(person_6)
                                            #print(input.shape)
                                            input6 = input6.astype('float32')
                                            listdata_frame_person6.append(input6)
                                            self.counter2_6=0
                                            test_video6 = np.array(listdata_frame_person6)
                                            num_of_samples6=len(test_video6)
                                            print('Number of samples5: ', num_of_samples6)
                                            print('data_shape5: ', test_video6.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.expand_dims(test_video6,axis=1)
                                                else:
                                                    test_video6=np.expand_dims(test_video6,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.rollaxis(test_video6,3,1)
                                                else:
                                                    test_video6=np.rollaxis(test_video6,3,4)
                                              
                                            #*****************************************
                                            print(test_video6.shape)                      
                                            test_video6=Normalize(test_video6)                     
                                            out_kelass47 = model27.predict_classes(test_video6)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass47==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass47==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass47==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass47==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_6 = []
                                            listdata_frame_person6 = []
                                        
                                        
                                        
                                    if mlb.classes_[j]=='yellow':
                                        self.counter2_6 += 1
                                        person_6.append(next)
                                        text_count=  "ID: 8"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_6==5:        
                                            input6 = np.array(person_6)
                                            #print(input.shape)
                                            input6 = input6.astype('float32')
                                            listdata_frame_person6.append(input6)
                                            self.counter2_6=0
                                            test_video6 = np.array(listdata_frame_person6)
                                            num_of_samples6=len(test_video6)
                                            print('Number of samples5: ', num_of_samples6)
                                            print('data_shape5: ', test_video6.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.expand_dims(test_video6,axis=1)
                                                else:
                                                    test_video6=np.expand_dims(test_video6,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.rollaxis(test_video6,3,1)
                                                else:
                                                    test_video6=np.rollaxis(test_video6,3,4)
                                              
                                            #*****************************************
                                            print(test_video6.shape)                      
                                            test_video6=Normalize(test_video6)                     
                                            out_kelass48 = model28.predict_classes(test_video6)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass48==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass48==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass48==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass48==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_6 = []
                                            listdata_frame_person6 = []
                                            
                                            
                                            
                                    if mlb.classes_[j]=='black':
                                        self.counter2_6 += 1
                                        person_6.append(next)
                                        text_count=  "ID: 9"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_6==5:        
                                            input6 = np.array(person_6)
                                            #print(input.shape)
                                            input6 = input6.astype('float32')
                                            listdata_frame_person6.append(input6)
                                            self.counter2_6=0
                                            test_video6 = np.array(listdata_frame_person6)
                                            num_of_samples6=len(test_video6)
                                            print('Number of samples5: ', num_of_samples6)
                                            print('data_shape5: ', test_video6.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.expand_dims(test_video6,axis=1)
                                                else:
                                                    test_video6=np.expand_dims(test_video6,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video6=np.rollaxis(test_video6,3,1)
                                                else:
                                                    test_video6=np.rollaxis(test_video6,3,4)
                                              
                                            #*****************************************
                                            print(test_video6.shape)                      
                                            test_video6=Normalize(test_video6)                     
                                            out_kelass71 = model29.predict_classes(test_video6)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass71==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass71==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass71==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass71==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_6 = []
                                            listdata_frame_person6 = []
                                            
                                            
                                        
                                        
                                        
                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                        
                        if last_item==7: # agar akharin tedade mashine k tashkhis dadi 2 bod?
                                #count2 = 2
                                #print(self.count2)
                                #text_count=  " " + ('%.d' %count2)
                                #cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                #if num_car[1]=='2': 
                                frame7 = cv2.resize(letter,(96,96))
                                frame7 = frame7.astype("float") / 255.0
                                input_data7 = img_to_array(frame7) 
                                input_data7 = np.expand_dims(input_data7, axis=0)
                                #print(input_data2.shape)
                                    
                                # labels 
                                #print("[INFO] classifying image...")
                                proba7 = model7.predict(input_data7)[0]
                                idxs7 = np.argsort(proba7)[::-1][:2]
                            
                                # loop over 
                                for (i, j) in enumerate(idxs7):
                                    # build the label and draw the label on the image
                                    label7 = "{}: {:.2f}%".format(mlb.classes_[j], proba7[j] * 100)
                                    cv2.putText(letter, label7, (385, (i * 30) + 25), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    if mlb.classes_[j]=='blue':
                                        self.counter2_7 += 1
                                        person_7.append(next)
                                        text_count=  "ID: 1"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_7==5:
                                            input7 = np.array(person_7)
                                            input7 = input7.astype('float32')
                                            listdata_frame_person7.append(input7)
                                            self.counter2_7=0
                                            test_video7 = np.array(listdata_frame_person7)
                                            num_of_samples7=len(test_video7)
                                            print('Number of samples: ', num_of_samples7)
                                            print('data_shape: ', test_video7.shape)
                                            
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.expand_dims(test_video7,axis=1)
                                                else:
                                                    test_video7=np.expand_dims(test_video7,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.rollaxis(test_video7,3,1)
                                                else:
                                                    test_video7=np.rollaxis(test_video7,3,4)
                                                                    
                                    #*************************************************     
                                            test_video7=Normalize(test_video7)
                                            out_kelass49 = model21.predict_classes(test_video7)
        
                                    #*************************************************
        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            if out_kelass49==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                                
                                            elif out_kelass49==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown_action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass49==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass49==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            person_7= []
                                            listdata_frame_person7 = []
                                            
                                    if mlb.classes_[j]=='brown':
                                        self.counter2_7 += 1
                                        person_7.append(next)
                                        text_count=  "ID: 2"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_7==5:      
                                            input7 = np.array(person_7)
                                            #print(input.shape)
                                            input7 = input7.astype('float32')
                                            listdata_frame_person7.append(input7)
                                            self.counter2_7=0
                                            test_video7 = np.array(listdata_frame_person7)
                                            num_of_samples7=len(test_video7)
                                            print('Number of samples2: ', num_of_samples7)
                                            print('data_shape2: ', test_video7.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.expand_dims(test_video7,axis=1)
                                                else:
                                                    test_video7=np.expand_dims(test_video7,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.rollaxis(test_video7,3,1)
                                                else:
                                                    test_video7=np.rollaxis(test_video7,3,4)
                                                    
                                            #****************************************
                                            print(test_video7.shape)                    
                                            test_video7=Normalize(test_video7)                     
                                            out_kelass50 = model22.predict_classes(test_video7)     
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************************************
                                            
                                            if out_kelass50==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)   
                                                
                                            elif out_kelass50==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass50==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass50==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_7 = []
                                            listdata_frame_person7 = []
                                            
                                    if mlb.classes_[j]=='green':
                                        self.counter2_7 += 1
                                        person_7.append(next)
                                        text_count=  "ID: 3"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_7==5:         
                                            input7 = np.array(person_7)
                                            #print(input.shape)
                                            input7 = input7.astype('float32')
                                            listdata_frame_person7.append(input7)
                                            self.counter2_7=0
                                            test_video7 = np.array(listdata_frame_person7)
                                            num_of_samples7=len(test_video7)
                                            print('Number of samples3: ', num_of_samples7)
                                            print('data_shape3: ', test_video7.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.expand_dims(test_video7,axis=1)
                                                else:
                                                    test_video7=np.expand_dims(test_video7,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.rollaxis(test_video7,3,1)
                                                else:
                                                    test_video7=np.rollaxis(test_video7,3,4)
                                                    
                                            #*****************************************
                                            print(test_video7.shape)                       
                                            test_video7=Normalize(test_video7)                     
                                            out_kelass51= model23.predict_classes(test_video7)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************** **********************                                   
                                            
                                            if out_kelass51==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass51==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass51==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass51==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_7 = []
                                            listdata_frame_person7 = []  
        
        
                                    if mlb.classes_[j]=='orange':
                                        self.counter2_7 += 1
                                        person_7.append(next)
                                        text_count=  "ID: 4"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_7==5:       
                                            input7 = np.array(person_7)
                                            #print(input.shape)
                                            input7 = input7.astype('float32')
                                            listdata_frame_person7.append(input7)
                                            self.counter2_7=0
                                            test_video7 = np.array(listdata_frame_person7)
                                            num_of_samples7=len(test_video7)
                                            print('Number of samples4: ', num_of_samples7)
                                            print('data_shape4: ', test_video7.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.expand_dims(test_video7,axis=1)
                                                else:
                                                    test_video7=np.expand_dims(test_video7,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.rollaxis(test_video7,3,1)
                                                else:
                                                    test_video7=np.rollaxis(test_video7,3,4)
                                                    
                                            #*****************************************
                                            print(test_video7.shape)                      
                                            test_video7=Normalize(test_video7)                
                                            out_kelass52 = model24.predict_classes(test_video7)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass52==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass52==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass52==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass52==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_7 = []
                                            listdata_frame_person7 = []
                                        
                                        
                                    if mlb.classes_[j]=='purple':
                                        self.counter2_7 += 1
                                        person_7.append(next)
                                        text_count=  "ID: 5"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_7==5:        
                                            input7 = np.array(person_7)
                                            #print(input.shape)
                                            input7 = input7.astype('float32')
                                            listdata_frame_person7.append(input7)
                                            self.counter2_7=0
                                            test_video7 = np.array(listdata_frame_person7)
                                            num_of_samples7=len(test_video7)
                                            print('Number of samples5: ', num_of_samples7)
                                            print('data_shape5: ', test_video7.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.expand_dims(test_video7,axis=1)
                                                else:
                                                    test_video7=np.expand_dims(test_video7,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.rollaxis(test_video7,3,1)
                                                else:
                                                    test_video7=np.rollaxis(test_video7,3,4)
                                              
                                            #*****************************************
                                            print(test_video7.shape)                      
                                            test_video7=Normalize(test_video7)                     
                                            out_kelass53 = model25.predict_classes(test_video7)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass53==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass53==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass53==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass53==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_7 = []
                                            listdata_frame_person7 = []                 
        
            
                                    if mlb.classes_[j]=='red':
                                        self.counter2_7 += 1
                                        person_7.append(next)
                                        #self.counter2 += 1
                                        #print(self.counter2)
                                        text_count=  "ID: 6"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_7==5:        
                                            input7 = np.array(person_7)
                                            #print(input.shape)
                                            input7 = input7.astype('float32')
                                            listdata_frame_person7.append(input7)
                                            self.counter2_7=0
                                            test_video7 = np.array(listdata_frame_person7)
                                            num_of_samples7=len(test_video7)
                                            print('Number of samples5: ', num_of_samples7)
                                            print('data_shape5: ', test_video7.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.expand_dims(test_video7,axis=1)
                                                else:
                                                    test_video7=np.expand_dims(test_video7,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.rollaxis(test_video7,3,1)
                                                else:
                                                    test_video7=np.rollaxis(test_video7,3,4)
                                              
                                            #*****************************************
                                            print(test_video7.shape)                      
                                            test_video7=Normalize(test_video7)                     
                                            out_kelass54 = model26.predict_classes(test_video7)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass54==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass54==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass54==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass54==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_7 = []
                                            listdata_frame_person7 = []
        
        
                                    if mlb.classes_[j]=='white':
                                        self.counter2_7 += 1
                                        person_7.append(next)
                                        text_count=  "ID: 7"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_7==5:        
                                            input7 = np.array(person_7)
                                            #print(input.shape)
                                            input7 = input7.astype('float32')
                                            listdata_frame_person7.append(input7)
                                            self.counter2_7=0
                                            test_video7 = np.array(listdata_frame_person7)
                                            num_of_samples7=len(test_video7)
                                            print('Number of samples5: ', num_of_samples7)
                                            print('data_shape5: ', test_video7.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.expand_dims(test_video7,axis=1)
                                                else:
                                                    test_video7=np.expand_dims(test_video7,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.rollaxis(test_video7,3,1)
                                                else:
                                                    test_video7=np.rollaxis(test_video7,3,4)
                                              
                                            #*****************************************
                                            print(test_video7.shape)                      
                                            test_video7=Normalize(test_video7)                     
                                            out_kelass55 = model27.predict_classes(test_video7)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass55==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass55==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass55==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass55==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_7 = []
                                            listdata_frame_person7 = []
                                        
                                        
                                        
                                    if mlb.classes_[j]=='yellow':
                                        self.counter2_7 += 1
                                        person_7.append(next)
                                        text_count=  "ID: 8"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_7==5:        
                                            input7 = np.array(person_7)
                                            #print(input.shape)
                                            input7 = input7.astype('float32')
                                            listdata_frame_person7.append(input7)
                                            self.counter2_7=0
                                            test_video7 = np.array(listdata_frame_person7)
                                            num_of_samples7=len(test_video7)
                                            print('Number of samples5: ', num_of_samples7)
                                            print('data_shape5: ', test_video7.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.expand_dims(test_video7,axis=1)
                                                else:
                                                    test_video7=np.expand_dims(test_video7,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.rollaxis(test_video7,3,1)
                                                else:
                                                    test_video7=np.rollaxis(test_video7,3,4)
                                              
                                            #*****************************************
                                            print(test_video7.shape)                      
                                            test_video7=Normalize(test_video7)                     
                                            out_kelass56 = model28.predict_classes(test_video7)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass56==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass56==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass56==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass56==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_7 = []
                                            listdata_frame_person7 = []
                                            
                                            
                                            
                                    if mlb.classes_[j]=='black':
                                        self.counter2_7 += 1
                                        person_7.append(next)
                                        text_count=  "ID: 9"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_7==5:        
                                            input7 = np.array(person_7)
                                            #print(input.shape)
                                            input7 = input7.astype('float32')
                                            listdata_frame_person7.append(input7)
                                            self.counter2_7=0
                                            test_video7 = np.array(listdata_frame_person7)
                                            num_of_samples7=len(test_video7)
                                            print('Number of samples5: ', num_of_samples7)
                                            print('data_shape5: ', test_video7.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.expand_dims(test_video7,axis=1)
                                                else:
                                                    test_video7=np.expand_dims(test_video7,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video7=np.rollaxis(test_video7,3,1)
                                                else:
                                                    test_video7=np.rollaxis(test_video7,3,4)
                                              
                                            #*****************************************
                                            print(test_video7.shape)                      
                                            test_video7=Normalize(test_video7)                     
                                            out_kelass72 = model29.predict_classes(test_video7)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass72==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass72==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass72==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass72==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_7 = []
                                            listdata_frame_person7 = []
                                            
                                        
                                        
                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                        
                        if last_item==8: # agar akharin tedade mashine k tashkhis dadi 2 bod?
                                #count2 = 2
                                #print(self.count2)
                                #text_count=  " " + ('%.d' %count2)
                                #cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                #if num_car[1]=='2': 
                                frame8 = cv2.resize(letter,(96,96))
                                frame8 = frame8.astype("float") / 255.0
                                input_data8 = img_to_array(frame8) 
                                input_data8 = np.expand_dims(input_data8, axis=0)
                                #print(input_data2.shape)
                                    
                                # labels 
                                #print("[INFO] classifying image...")
                                proba8 = model8.predict(input_data8)[0]
                                idxs8 = np.argsort(proba8)[::-1][:2]
                            
                                # loop over 
                                for (i, j) in enumerate(idxs8):
                                    # build the label and draw the label on the image
                                    label8 = "{}: {:.2f}%".format(mlb.classes_[j], proba8[j] * 100)
                                    cv2.putText(letter, label8, (385, (i * 30) + 25), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    if mlb.classes_[j]=='blue':
                                        self.counter2_8 += 1
                                        person_8.append(next)
                                        text_count=  "ID: 1"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_8==5:
                                            input8 = np.array(person_8)
                                            input8 = input8.astype('float32')
                                            listdata_frame_person8.append(input8)
                                            self.counter2_8=0
                                            test_video8 = np.array(listdata_frame_person8)
                                            num_of_samples8=len(test_video8)
                                            print('Number of samples: ', num_of_samples8)
                                            print('data_shape: ', test_video8.shape)
                                            
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.expand_dims(test_video8,axis=1)
                                                else:
                                                    test_video8=np.expand_dims(test_video8,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.rollaxis(test_video8,3,1)
                                                else:
                                                    test_video8=np.rollaxis(test_video8,3,4)
                                                                    
                                    #*************************************************     
                                            test_video8=Normalize(test_video8)
                                            out_kelass57 = model21.predict_classes(test_video8)
        
                                    #*************************************************
        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            if out_kelass57==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                                
                                            elif out_kelass57==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown_action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass57==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            elif out_kelass57==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving' , text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
        
                                            person_8= []
                                            listdata_frame_person8 = []
                                            
                                    if mlb.classes_[j]=='brown':
                                        self.counter2_8 += 1
                                        person_8.append(next)
                                        text_count=  "ID: 2"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_8==5:      
                                            input8 = np.array(person_8)
                                            #print(input.shape)
                                            input8 = input8.astype('float32')
                                            listdata_frame_person8.append(input8)
                                            self.counter2_8=0
                                            test_video8 = np.array(listdata_frame_person8)
                                            num_of_samples8=len(test_video8)
                                            print('Number of samples2: ', num_of_samples8)
                                            print('data_shape2: ', test_video8.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.expand_dims(test_video8,axis=1)
                                                else:
                                                    test_video8=np.expand_dims(test_video8,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.rollaxis(test_video8,3,1)
                                                else:
                                                    test_video8=np.rollaxis(test_video8,3,4)
                                                    
                                            #****************************************
                                            print(test_video8.shape)                    
                                            test_video8=Normalize(test_video8)                     
                                            out_kelass58 = model22.predict_classes(test_video8)     
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************************************
                                            
                                            if out_kelass58==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)   
                                                
                                            elif out_kelass58==[1]:
                                                print('unknown_action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass58==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass58==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_8 = []
                                            listdata_frame_person8 = []
                                            
                                    if mlb.classes_[j]=='green':
                                        self.counter2_8 += 1
                                        person_8.append(next)
                                        text_count=  "ID: 3"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_8==5:         
                                            input8 = np.array(person_8)
                                            #print(input.shape)
                                            input8 = input8.astype('float32')
                                            listdata_frame_person8.append(input8)
                                            self.counter2_8=0
                                            test_video8 = np.array(listdata_frame_person8)
                                            num_of_samples8=len(test_video8)
                                            print('Number of samples3: ', num_of_samples8)
                                            print('data_shape3: ', test_video8.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.expand_dims(test_video8,axis=1)
                                                else:
                                                    test_video8=np.expand_dims(test_video8,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.rollaxis(test_video8,3,1)
                                                else:
                                                    test_video8=np.rollaxis(test_video8,3,4)
                                                    
                                            #*****************************************
                                            print(test_video8.shape)                       
                                            test_video8=Normalize(test_video8)                     
                                            out_kelass59= model23.predict_classes(test_video8)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #****************** **********************                                   
                                            
                                            if out_kelass59==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass59==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass59==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass59==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_8 = []
                                            listdata_frame_person8 = []  
        
        
                                    if mlb.classes_[j]=='orange':
                                        self.counter2_8 += 1
                                        person_8.append(next)
                                        text_count=  "ID: 4"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_8==5:       
                                            input8 = np.array(person_8)
                                            #print(input.shape)
                                            input8 = input8.astype('float32')
                                            listdata_frame_person8.append(input8)
                                            self.counter2_8=0
                                            test_video8 = np.array(listdata_frame_person8)
                                            num_of_samples8=len(test_video8)
                                            print('Number of samples4: ', num_of_samples8)
                                            print('data_shape4: ', test_video8.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.expand_dims(test_video8,axis=1)
                                                else:
                                                    test_video8=np.expand_dims(test_video8,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.rollaxis(test_video8,3,1)
                                                else:
                                                    test_video8=np.rollaxis(test_video8,3,4)
                                                    
                                            #*****************************************
                                            print(test_video8.shape)                      
                                            test_video8=Normalize(test_video8)                
                                            out_kelass60 = model24.predict_classes(test_video8)        
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass60==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass60==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass60==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass60==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_8 = []
                                            listdata_frame_person8 = []
                                        
                                        
                                    if mlb.classes_[j]=='purple':
                                        self.counter2_8 += 1
                                        person_8.append(next)
                                        text_count=  "ID: 5"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_8==5:        
                                            input8 = np.array(person_8)
                                            #print(input.shape)
                                            input8 = input8.astype('float32')
                                            listdata_frame_person8.append(input8)
                                            self.counter2_8=0
                                            test_video8 = np.array(listdata_frame_person8)
                                            num_of_samples8=len(test_video8)
                                            print('Number of samples5: ', num_of_samples8)
                                            print('data_shape5: ', test_video8.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.expand_dims(test_video8,axis=1)
                                                else:
                                                    test_video8=np.expand_dims(test_video8,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.rollaxis(test_video8,3,1)
                                                else:
                                                    test_video8=np.rollaxis(test_video8,3,4)
                                              
                                            #*****************************************
                                            print(test_video8.shape)                      
                                            test_video8=Normalize(test_video8)                     
                                            out_kelass61 = model25.predict_classes(test_video8)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass61==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass61==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass61==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass61==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_8 = []
                                            listdata_frame_person8 = []                 
        
            
                                    if mlb.classes_[j]=='red':
                                        self.counter2_8 += 1
                                        person_8.append(next)
                                        #self.counter2 += 1
                                        #print(self.counter2)
                                        text_count=  "ID: 6"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)  
                                        if self.counter2_8==5:        
                                            input8 = np.array(person_8)
                                            #print(input.shape)
                                            input8 = input8.astype('float32')
                                            listdata_frame_person8.append(input8)
                                            self.counter2_8=0
                                            test_video8 = np.array(listdata_frame_person8)
                                            num_of_samples8=len(test_video8)
                                            print('Number of samples5: ', num_of_samples8)
                                            print('data_shape5: ', test_video8.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.expand_dims(test_video8,axis=1)
                                                else:
                                                    test_video8=np.expand_dims(test_video8,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.rollaxis(test_video8,3,1)
                                                else:
                                                    test_video8=np.rollaxis(test_video8,3,4)
                                              
                                            #*****************************************
                                            print(test_video8.shape)                      
                                            test_video8=Normalize(test_video8)                     
                                            out_kelass62 = model26.predict_classes(test_video8)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass62==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass62==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass62==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass62==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_8 = []
                                            listdata_frame_person8 = []
        
        
                                    if mlb.classes_[j]=='white':
                                        self.counter2_8 += 1
                                        person_8.append(next)
                                        text_count=  "ID: 7"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_8==5:        
                                            input8 = np.array(person_8)
                                            #print(input.shape)
                                            input8 = input8.astype('float32')
                                            listdata_frame_person8.append(input8)
                                            self.counter2_8=0
                                            test_video8 = np.array(listdata_frame_person8)
                                            num_of_samples8=len(test_video8)
                                            print('Number of samples5: ', num_of_samples8)
                                            print('data_shape5: ', test_video8.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.expand_dims(test_video8,axis=1)
                                                else:
                                                    test_video8=np.expand_dims(test_video8,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.rollaxis(test_video8,3,1)
                                                else:
                                                    test_video8=np.rollaxis(test_video8,3,4)
                                              
                                            #*****************************************
                                            print(test_video8.shape)                      
                                            test_video8=Normalize(test_video8)                     
                                            out_kelass63 = model27.predict_classes(test_video8)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass63==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass63==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass63==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass63==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_8 = []
                                            listdata_frame_person8 = []
                                        
                                        
                                        
                                    if mlb.classes_[j]=='yellow':
                                        self.counter2_8 += 1
                                        person_8.append(next)
                                        text_count=  "ID: 8"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_8==5:        
                                            input8 = np.array(person_8)
                                            #print(input.shape)
                                            input8 = input8.astype('float32')
                                            listdata_frame_person8.append(input8)
                                            self.counter2_8=0
                                            test_video8 = np.array(listdata_frame_person8)
                                            num_of_samples8=len(test_video8)
                                            print('Number of samples5: ', num_of_samples8)
                                            print('data_shape5: ', test_video8.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.expand_dims(test_video8,axis=1)
                                                else:
                                                    test_video8=np.expand_dims(test_video8,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.rollaxis(test_video8,3,1)
                                                else:
                                                    test_video8=np.rollaxis(test_video8,3,4)
                                              
                                            #*****************************************
                                            print(test_video8.shape)                      
                                            test_video8=Normalize(test_video8)                     
                                            out_kelass64 = model28.predict_classes(test_video8)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass64==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass64==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass64==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass64==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_8 = []
                                            listdata_frame_person8 = []  
                                            
                                            
                                            
                                            
                                    if mlb.classes_[j]=='black':
                                        self.counter2_8 += 1
                                        person_8.append(next)
                                        text_count=  "ID: 9"
                                        cv2.putText(to_draw, text_count, text_pos_counter, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                                        if self.counter2_8==5:        
                                            input8 = np.array(person_8)
                                            #print(input.shape)
                                            input8 = input8.astype('float32')
                                            listdata_frame_person8.append(input8)
                                            self.counter2_8=0
                                            test_video8 = np.array(listdata_frame_person8)
                                            num_of_samples8=len(test_video8)
                                            print('Number of samples5: ', num_of_samples8)
                                            print('data_shape5: ', test_video8.shape)
                                                
                                            if num_channel==1:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.expand_dims(test_video8,axis=1)
                                                else:
                                                    test_video8=np.expand_dims(test_video8,axis=4)    
                                            else:
                                                if t.image_data_format()=='channels_first':
                                                    test_video8=np.rollaxis(test_video8,3,1)
                                                else:
                                                    test_video8=np.rollaxis(test_video8,3,4)
                                              
                                            #*****************************************
                                            print(test_video8.shape)                      
                                            test_video8=Normalize(test_video8)                     
                                            out_kelass73 = model29.predict_classes(test_video8)         
                                            text_pos_label = (xmin + 5, ymin - 20)
                                            #*****************************************
                                            
                                            if out_kelass73==[0]:
                                                print('boxing')
                                                cv2.putText(to_draw, 'boxing', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                
                                            elif out_kelass73==[1]:
                                                print('unknown action')
                                                cv2.putText(to_draw, 'unknown action', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
            
                                            elif out_kelass73==[2]:
                                                print('walking')
                                                cv2.putText(to_draw, 'walking', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                
                                            elif out_kelass73==[3]:
                                                print('waving')
                                                cv2.putText(to_draw, 'waving', text_pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)                  
                                            person_8 = []
                                            listdata_frame_person8 = [] 
                                        
                                        
                                        
                                        
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        
                cv2.imshow("SSD result", to_draw)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        vid.release()
        cv2.destroyAllWindows()
            
            
        
