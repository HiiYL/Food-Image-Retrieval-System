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

from cyvlfeat import sift as cysift
from sklearn.mixture import GMM

# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
dbpath = 'fooddb/'   

# these labels are the abbreviations of the actual food names
labels = ('AK','BL','CD','CL','DR','MG','NL','PG','RC','ST')
    
featvect = []  # empty list for holding features
FEtime = np.zeros(1000)

def extract_image_features(image):
    _ , descriptors =  cysift.sift(img_gray, peak_thresh=8, edge_thresh=5, compute_descriptor=True)
    return descriptors

def dictionary(descriptors, N):
    em = cv2.EM(N)
    em.train(descriptors)

    return np.float32(em.getMat("means")), \
        np.float32(em.getMatVector("covs")), np.float32(em.getMat("weights"))[0]

def generate_gmm(image_features_list):
    concatenated_descriptors = np.concatenate(image_features_list)
    gmm_filename = 'gmm.pkl'
    N, D = concatenated_descriptors.shape
    K=128

    print("The sizes are {0} and {1}".format(N,D))

    if(N > 3000000):
        batch_size = 3000000
    else:
        batch_size = N

    # ggmm.init(batch_size * D)
    # gmm = ggmm.GMM(K,D)

    thresh = 1e-3 # convergence threshold
    n_iter = 500 # maximum number of EM iterations
    init_params = 'wmc' # initialize weights, means, and covariances

    # gmm = GMM(n_components=10,
    #                 covariance_type='spherical', init_params='wc', n_iter=20)

    # train GMM
    # converged = gmm.fit(concatenated_descriptors[:batch_size], thresh, n_iter, init_params=init_params)

    em = cv2.ml.EM_create()
    em.setClustersNumber(K)
    em.trainEM(concatenated_descriptors)


    print("GMM converged? ... {0}".format(converged))

    return gmm

def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.compute_posteriors(xx)  # NxK
    
    Q = Q.asarray()

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.get_weights()
    d_mu = Q_xx - Q_sum * gmm.get_means()
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.get_means() ** 2
        + Q_sum * gmm.get_covars()
        + 2 * Q_xx * gmm.get_means())

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

image_features_list = []
for idx in range(1000):
    img = cv2.imread( os.path.join(dbpath, str(idx) + ".jpg") )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # display image
    #plt.imshow(img), plt.xticks([]), plt.yticks([])
    #plt.show()
    
    # compute features and append to list
    e1 = cv2.getTickCount() # start timer
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    descriptors = extract_image_features(img_gray)

    if descriptors is not None and descriptors.shape[0] >= 64:
        image_features_list.append(descriptors)
    # feat = computeFeatures(img, deep_features=False)
    e2 = cv2.getTickCount()  # stop timer

    FEtime[idx] = (e2 - e1) / cv2.getTickFrequency() 
    
    print('Extracting features for image #%d'%idx )


print("Generating GMM")
gmm = generate_gmm(image_features_list)
fv = [ fisher_vector(image,gmm) for image in image_features_list]