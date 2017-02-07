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
fv = pickle.load(open("feat.pkl", "rb") )
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
nRetrieved = 50

# access distances of all images from query image (first image), sort them asc
nearest_idx = np.argsort(D[0, :]);

# quick way of finding category label for top K retrieved images
retrievedCats = np.uint8(np.floor((nearest_idx[1:nRetrieved+1])/100));

retrievedCats = np.array(retrievedCats)
 
# find matches
hits_q = (retrievedCats == gt_idx)

# print(hits_binary)

print("List of hits : {}".format(list(map(int, hits_q))))

print("Average Precision (Mine): %.4f"%(average_precision(hits_q)))


# from sklearn.metrics import average_precision_score
# print(average_precision_score([ gt_idx ] * nRetrieved, retrievedCats))

# print([ gt_idx ] * nRetrieved)
# print(retrievedCats)

# print(hits_q/(np.arange(nRetrieved)+1))
# print((np.arange(nRetrieved)+1))
# print(np.sum(hits_q/(np.arange(nRetrieved)+1)))
# print(np.sum(hits_q))
  
# calculate average precision of the ranked matches
# print(retrievedCats)

# precision_at_each_step = [ (np.count_nonzero(retrievedCats[:i] == gt_idx) / i) for i in range(1,nRetrieved+1) if (retrievedCats[i-1] == gt_idx) ]

# print(np.average(precision_at_each_step) )
if np.sum(hits_q) != 0:
  avg_prec_q = np.sum(hits_q/(np.arange(nRetrieved)+1)) / np.sum(hits_q)
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
    
plt.show()