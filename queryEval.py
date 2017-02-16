"""
queryEval.py

DO NOT MODIFY ANY CODES IN THIS FILE
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 

@author: John See, 2017
For questions or bug reporting, please send an email to johnsee@mmu.edu.my

"""
import os
import cv2
import numpy as np
import pickle
import sys, getopt
import matplotlib.pyplot as plt
from computeFeatures import computeFeatures
from computeDistances import computeDistances
from itertools import cycle

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
dbpath = 'fooddb/'  

# these labels are the abbreviations of the actual food names
labels = ('AK','BL','CD','CL','DR','MG','NL','PG','RC','ST')

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"d:q:h")

# parsing command line args
for o, a in myopts:
    if o == '-d':
        queryfile = os.path.join(dbpath, a + '.jpg')
        gt_idx = np.uint8(np.floor(int(a)/100))
        if not os.path.isfile(queryfile):
            print("Error: Query file does not exist! Please check.")
            sys.exit()
    elif o == '-q':
        queryfile = a
        if not os.path.isfile(queryfile):
            print("Error: Query file does not exist! Please check.")
            sys.exit()
        # tokenize filename to get category label and index
        gt = str(queryfile.split("_")[1]).split(".")[0]
        gt_idx = labels.index(gt)
    elif o == '-h':
        print("\nUsage: %s -d dbfilenumber\n       # to specify a single query image from the database for evaluation" % sys.argv[0])
        print("\n       %s -q queryfile\n       # to specify a new query image for evaluation" % sys.argv[0])
        print(" ")       
        sys.exit()
    else:
        print(' ')
    

featvect = []  # empty list for holding features
FEtime = np.zeros(1000)

# load pickled features
fv = pickle.load(open("feat-deep.pkl", "rb") )
print('Features loaded')

# read query image file
img = cv2.imread(queryfile)
query_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# show stuff
plt.imshow(query_img), plt.title('Query image: %s'%labels[gt_idx])
plt.xticks([]), plt.yticks([])
print('Query image: %s'%labels[gt_idx])

# compute features
newfeat = computeFeatures(query_img)

# insert new feat to the top of the feature vector stack
fv = np.insert(fv, 0, newfeat, axis=0)

# find all pairwise distances
D = computeDistances(fv)


# *** Evaluation ----------------------------------------------------------

# number of images to retrieve
nRetrieved = 100

# access distances of all images from query image (first image), sort them asc
nearest_idx = np.argsort(D[0, :]);

# quick way of finding category label for top K retrieved images
retrievedCats = np.uint8(np.floor((nearest_idx[1:nRetrieved+1])/100));

retrievedCats = np.array(retrievedCats)
 
# find matches
hits_q = (retrievedCats == gt_idx)

if np.sum(hits_q) != 0:
  avg_prec_q = np.sum(hits_q*np.cumsum(hits_q)/(np.arange(nRetrieved)+1)) / np.sum(hits_q)
else:
  avg_prec_q = 0.0
          
recall = np.sum(hits_q) / nRetrieved

# *** Results & Visualization-----------------------------------------------

print('Average Precision, AP@%d: %.4f'%(nRetrieved,avg_prec_q))
print('Recall Rate@%d: %.4f'%(nRetrieved,recall)) 

fig, axs = plt.subplots(2, 5, figsize=(15, 6), facecolor='w', edgecolor='w')
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
for i in range(10):
    imgfile = os.path.join(dbpath, str(nearest_idx[i+1]) + '.jpg')
    matched_img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)
    axs[i].imshow(matched_img)
    axs[i].set_title(str(i+1) + '. ' + labels[retrievedCats[i]])
    axs[i].set_xticks([])
    axs[i].set_yticks([])

# precision_array = np.zeros(nRetrieved)
# recall_array = np.zeros(nRetrieved)

# for i in range(nRetrieved):
#     current_hits_q = hits_q[:i]
#     sum_relevant = np.sum(current_hits_q)

#     precision_array[i] = np.sum(current_hits_q*np.cumsum(current_hits_q)/(np.arange(i)+1)) / np.sum(current_hits_q)# / np.sum(current_hits_q)
#     recall_array[nRetrieved-i-1] = sum_relevant / i

# # setup plot details
# colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
# lw = 2

# plt.clf()
# plt.plot(recall_array, precision_array, lw=lw, color='navy',
#          label='Precision-Recall curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('Precision-Recall Curve')
# plt.legend(loc="lower left")
    
plt.show()