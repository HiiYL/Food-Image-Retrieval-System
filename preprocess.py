"""
featureExtraction.py

DO NOT MODIFY ANY CODES IN THIS FILE
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 

@author: John See, 2017
For questions or bug reporting, please send an email to johnsee@mmu.edu.my

"""
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from computeFeatures import computeFeatures


from squeezenet import get_squeezenet
from keras.optimizers import SGD, RMSprop,Adam

from keras.utils.np_utils import to_categorical

width = 227
height = 227
channel = 3

imagesCount = 1000


def preprocessImage(image):
    im = cv2.resize(image, (width, height)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im


squeezenet_model = get_squeezenet(nb_classes=10,
 path_to_weights='model/squeezenet_weights_th_dim_ordering_th_kernels.h5',
 dim_ordering='th')

adam = Adam(lr=0.0001,clipnorm=1.,clipvalue=0.5)
squeezenet_model.compile(optimizer=adam,loss='categorical_crossentropy', metrics=['accuracy'])#,loss_weights=[1., 0.2])

# from keras.utils.visualize_util import plot
# plot(squeezenet_model, to_file='{}.png'.format('squeezenet'),show_shapes=True)

squeezenet_model.fit(data, Y_train, nb_epoch=5, batch_size=32)

# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
dbpath = 'fooddb/'  

# these labels are the abbreviations of the actual food names
labels = ('AK','BL','CD','CL','DR','MG','NL','PG','RC','ST')

y = np.array([ [i] * 100 for i in range(10)]).flatten()

Y_train = to_categorical(y, 10)





featvect = []  # empty list for holding features
# FEtime = np.zeros(1000)

if os.path.isfile('images.npy'):
    data = np.load('images.npy')
else:
    data = np.empty([imagesCount, channel, width,height])

    for idx in range(imagesCount):
        img = cv2.imread( os.path.join(dbpath, str(idx) + ".jpg") )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print('Extracting features for image #%d'%idx )

        data[idx] = preprocessImage(img)

    np.save('images.npy' ,data)

e1 = cv2.getTickCount()
featvect = squeezenet_model.predict(data)
e2 = cv2.getTickCount()

FEtime = (e2 - e1) /cv2.getTickFrequency() 
print('Feature extraction runtime: %.4f seconds'%np.sum(FEtime))



    
    # display image
    #plt.imshow(img), plt.xticks([]), plt.yticks([])
    #plt.show()
    
    # # compute features and append to list
    # e1 = cv2.getTickCount() # start timer
    # feat = computeFeatures(img, model=squeezenet_model)
    # e2 = cv2.getTickCount()  # stop timer
    
    # featvect.append( feat ); 
    # FEtime[idx] = (e2 - e1) / cv2.getTickFrequency() 
    
    # print('Extracting features for image #%d'%idx )



# temparr = np.array(featvect)
# fv = np.reshape(temparr, (temparr.shape[0], temparr.shape[1]) )
# del temparr

# pickle your features
pickle.dump( featvect, open( "feat.pkl", "wb" ) )
print('Features pickled!')
