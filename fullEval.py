"""
fullEval.py

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
from computeDistances import computeDistances

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    # print(r.size)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / r.size #np.mean(out)

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != False
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


# Defaults
dbSize = 1000       # number of images in food database
nPerCat = 100       # number of images in food database for each category
nC = 10             # number of categories
nRetrieved = 50     # number of images to retrieve
loadFV = True       # flag to indicate if feature vector will be loaded

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"r:th")

# parsing command line args
for o, a in myopts:
    if o == '-r':
        nRetrieved = int(a)
        if (nRetrieved > dbSize):
            print("Error: Number of retrieved images exceeds size of database!")
            sys.exit()
    elif o == '-t':          # extract features before evaluating
        cont = input('Caution! Do you wish to continue with feature extraction? (y/n): ')    
        if (cont == 'y'):
            exec(open("featureExtraction.py").read())
            loadFV = False
            print('Done extracting')
        else:
            print("\nCommand aborted. Start over again.")
            sys.exit()
    elif o == '-h':
        print("\nUsage: %s -r numRetrieved    # to specify number of retrieved images" % sys.argv[0])
        print("\n       %s -t         # to enable feature extraction before evaluation" % sys.argv[0])
        print(" ")       
        sys.exit()
    else:
        print(' ')

if loadFV:
    # load pickled features
    fv = pickle.load(open("feat.pkl", "rb") )
    print('Features loaded')
        
    
# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
dbpath = 'fooddb/'   

# these labels are the abbreviations of the actual food names
labels = ('AK','BL','CD','CL','DR','MG','NL','PG','RC','ST')

featvect = []  # empty list for holding features
FEtime = np.zeros(dbSize)

# find all pairwise distances
D = computeDistances(fv)


# *** Evaluation ----------------------------------------------------------
avg_prec = np.zeros(dbSize)


# iterate through all images from each category as query image
for c in range(nC): 
  for i in range(nPerCat):
      idx = (c*nPerCat) + i;

      # access distances of all images from query image, sort them asc
      nearest_idx = np.argsort(D[idx, :]);

      # quick way of finding category label for top K retrieved images
      retrievedCats = np.uint8(np.floor((nearest_idx[1:nRetrieved+1])/nPerCat));
 
      # find matches
      hits = (retrievedCats == np.floor(idx/nPerCat))
      
      # calculate average precision of the ranked matches
      if np.sum(hits) != 0:
          avg_prec[idx] = average_precision(hits)
      else:
          avg_prec[idx] = 0.0
          
mean_avg_prec = np.mean(avg_prec)
mean_avg_prec_perCat = np.mean(avg_prec.reshape(nPerCat, nC), axis=0)
recall = np.sum(hits) / nPerCat

# *** Results & Visualization-----------------------------------------------

print('Mean Average Precision, MAP@%d: %.4f'%(nRetrieved,mean_avg_prec))
print('Recall Rate@%d: %.4f'%(nRetrieved,recall)) 

x = np.arange(nC)+0.5
plt.xticks(x, list(labels) )
plt.xlim([0,10]), plt.ylim([0,1])
markerline, stemlines, baseline = plt.stem(x, mean_avg_prec_perCat, '-.')
plt.grid(True)
plt.xlabel('Food categories'), plt.ylabel('MAP per category')

#fig, axs = plt.subplots(2, 5, figsize=(12, 6), facecolor='w', edgecolor='w')
#fig.subplots_adjust(hspace = .5, wspace=.001)
#axs = axs.ravel()
#for i in range(nC):
#    imgfile = os.path.join(dbpath, str(nearest_idx[i+1]) + '.jpg')
#    matched_img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)
#    axs[i].imshow(matched_img)
#    axs[i].set_title(str(i+1) + '. ' + labels[retrievedCats[i]])
#    axs[i].set_xticks([])
#    axs[i].set_yticks([])

plt.show()
      