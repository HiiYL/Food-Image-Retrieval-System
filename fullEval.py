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

from itertools import cycle

# Defaults
dbSize = 1000       # number of images in food database
nPerCat = 100       # number of images in food database for each category
nC = 10             # number of categories
nRetrieved = 100     # number of images to retrieve
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
    fv = pickle.load(open("feat-deep.pkl", "rb") )
    fv_baseline = pickle.load(open("feat.pkl", "rb") )
    print('Features loaded')
        
    
# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
dbpath = 'fooddb/'   

# these labels are the abbreviations of the actual food names
labels = ('AK','BL','CD','CL','DR','MG','NL','PG','RC','ST')

featvect = []  # empty list for holding features
FEtime = np.zeros(dbSize)

# find all pairwise distances
D = computeDistances(fv)
D_baseline = computeDistances(fv_baseline)


# *** Evaluation ----------------------------------------------------------
def evaluate(D):
  avg_prec = np.zeros(dbSize)
  retrieves = range(1,1000,10)
  precisions = []
  recalls = []
  for retrive in retrieves:
    for c in range(nC): 
      for i in range(nPerCat):
          idx = (c*nPerCat) + i;
    
          # access distances of all images from query image, sort them asc
          nearest_idx = np.argsort(D[idx, :]);
    
          # quick way of finding category label for top K retrieved images
          retrievedCats = np.uint8(np.floor((nearest_idx[1:retrive+1])/nPerCat));
          
          # find matches
          hits = (retrievedCats == np.floor(idx/nPerCat))
          
          # calculate average precision of the ranked matches
          if np.sum(hits) != 0:
              avg_prec[idx] = np.sum(hits*np.cumsum(hits)/(np.arange(retrive)+1)) / np.sum(hits)
          else:
              avg_prec[idx] = 0.0
          
    mean_avg_prec = np.mean(avg_prec)
    recall = np.sum(hits) / nPerCat
    
    precisions = precisions + [mean_avg_prec]
    recalls = recalls + [recall]
  return precisions, recalls


precisions,recalls = evaluate(D)
precisions_baseline,recalls_baseline = evaluate(D_baseline)


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
          avg_prec[idx] = np.sum(hits*np.cumsum(hits)/(np.arange(nRetrieved)+1)) / np.sum(hits)#average_precision(hits)
      else:
          avg_prec[idx] = 0.0
          
mean_avg_prec = np.mean(avg_prec)
mean_avg_prec_perCat = np.mean(avg_prec.reshape(nPerCat, nC), axis=0)
recall = np.sum(hits) / nPerCat



# *** Results & Visualization-----------------------------------------------

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2

plt.clf()
plt.plot(recalls, precisions, lw=lw, color='navy',
         label='Squeezenet')
plt.plot(recalls_baseline, precisions_baseline, lw=lw, color='darkorange',
         label='Color Histogram')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.figure()

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
      