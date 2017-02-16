## Hack to modify dim ordering
import os

filename = os.path.join(os.path.expanduser('~'), '.keras', 'keras.json')
if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

with open(filename, "w") as f:
    f.write('{"backend": "theano","floatx": "float32","epsilon": 1e-07,"image_dim_ordering": "th"}')

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


# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
dbpath = 'fooddb/'  

# these labels are the abbreviations of the actual food names
labels = ('AK','BL','CD','CL','DR','MG','NL','PG','RC','ST')


featvect = []  # empty list for holding features
# FEtime = np.zeros(1000)

print("[INFO] Checking for image data")
if os.path.isfile('images.npy'):
    data = np.load('images.npy')
    print("[INFO] Preprocessed image data loaded")
else:
    print("[INFO] Image data not found, will now attempt to preprocess images from dbpath directory")
    data = np.empty([imagesCount, channel, width,height])

    for idx in range(imagesCount):
        img = cv2.imread( os.path.join(dbpath, str(idx) + ".jpg") )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print('Preprocessing image data for image #%d'%idx )
        data[idx] = preprocessImage(img)

    np.save('images.npy' ,data)

if not os.path.isfile('model/squeezenet_food.h5'):
    print("[WARN] Squeezenet finetuned weights not found, will now perform finetuning on dataset")
    print("[NOTE] If this happens for submitted code, something has gone terribly wrong")
    y = np.array([ [i] * 100 for i in range(10)]).flatten()
    Y_train = to_categorical(y, 10)
    squeezenet_model = get_squeezenet(nb_classes=10,
     path_to_weights='model/squeezenet_weights_th_dim_ordering_th_kernels.h5',
     dim_ordering='th')

    adam = Adam(lr=0.0001,clipnorm=1.,clipvalue=0.5)
    squeezenet_model.compile(optimizer=adam,loss='categorical_crossentropy', metrics=['accuracy'])#,loss_weights=[1., 0.2])

    # from keras.utils.visualize_util import plot
    # plot(squeezenet_model, to_file='{}.png'.format('squeezenet'),show_shapes=True)

    squeezenet_model.fit(data, Y_train, nb_epoch=10, batch_size=32)
    squeezenet_model.save_weights('model/squeezenet_food.h5')

else:
    squeezenet_model = get_squeezenet(nb_classes=10,
     path_to_weights='model/squeezenet_food.h5',
     dim_ordering='th')
    print("[INFO] Squeezenet model loaded with finetuned weights")

print('[INFO] Performing feedforward on network to extract features ...')
e1 = cv2.getTickCount()
featvect = squeezenet_model.predict(data)
e2 = cv2.getTickCount()

FEtime = (e2 - e1) /cv2.getTickFrequency() 
print('[INFO] Feature extraction runtime: %.4f seconds'%np.sum(FEtime))

# pickle your features
print('[INFO] Saving features to pickle...')
pickle.dump( featvect, open( "feat-deep.pkl", "wb" ) )
print('[INFO] Features pickled!')
