import skimage.io as io
from skimage.transform import  rescale,resize
from skimage.util import img_as_uint,img_as_ubyte
from skimage.color import rgb2gray
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from utils import *
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.utils import plot_model

def extractFeaturesTest(pre_image):
    feature_pool=np.empty([1,252])
    img = pre_image
    img_rescaled=(img-np.min(img))/(np.max(img)-np.min(img)) 
    
    texture_features=compute_14_features(img_rescaled)#texture features
    
    fft_map=np.fft.fft2(img_rescaled)
    fft_map = np.fft.fftshift(fft_map)
    fft_map = np.abs(fft_map)
    YC=int(np.floor(fft_map.shape[1]/2)+1)
    fft_map=fft_map[:,YC:int(np.floor(3*YC/2))]
    fft_features=compute_14_features(fft_map)#FFT features
    
    wavelet_coeffs = pywt.dwt2(img_rescaled,'sym4')
    cA1, (cH1, cV1, cD1) = wavelet_coeffs
    wavelet_coeffs = pywt.dwt2(cA1,'sym4')
    cA2, (cH2, cV2, cD2) = wavelet_coeffs#wavelet features
    wavelet_features=np.concatenate((compute_14_features(cA1), compute_14_features(cH1),compute_14_features(cV1),compute_14_features(cD1)
    ,compute_14_features(cA2), compute_14_features(cH2),compute_14_features(cV2),compute_14_features(cD2)), axis=0)
    
    
    gLDM1,gLDM2,gLDM3,gLDM4=GLDM(img_rescaled,10)#GLDM in four directions
    gldm_features=np.concatenate((compute_14_features(gLDM1), compute_14_features(gLDM2),
                                  compute_14_features(gLDM3),compute_14_features(gLDM4)), axis=0)
    
    
    glcms =greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])#GLCM in four directions
    glcm_features=np.concatenate((compute_14_features(im2double(glcms[:, :, 0, 0])), 
                                  compute_14_features(im2double(glcms[:, :, 0, 1])),
                                  compute_14_features(im2double(im2double(glcms[:, :, 0, 2]))),
                                  compute_14_features(glcms[:, :, 0, 3])), axis=0)
    
    feature_vector=np.concatenate((texture_features,fft_features,wavelet_features,gldm_features,glcm_features), axis=0).reshape(1,252)#merge to create a feature vector of 252
    feature_pool=np.concatenate((feature_pool,feature_vector), axis=0)
    feature_pool=np.delete(feature_pool, 0, 0)
    feature_pool=np.concatenate((feature_pool,0*np.ones(len(feature_pool)).reshape(len(feature_pool),1)), axis=1)#add label to the last column   
    sio.savemat('testFeatures.mat', {'testFeatures': feature_pool})

#extractFeaturesTest()
    source_dir = './'
    test_features=sio.loadmat(os.path.join(source_dir,'testFeatures.mat')) 
    test_features = test_features['testFeatures']

    X_test=test_features[:,:-1] #inputs

    covid_features=sio.loadmat(os.path.join(source_dir,'covid.mat')) 
    covid_features=covid_features['covid'] 

    normal_features=sio.loadmat(os.path.join(source_dir,'normal.mat')) 
    normal_features=normal_features['normal']  

    pneumonia_features=sio.loadmat(os.path.join(source_dir,'pneumonia.mat')) 
    pneumonia_features=pneumonia_features['pneumonia']  
    X=np.concatenate((covid_features[:,:-1],normal_features[:,:-1],pneumonia_features[:,:-1]), axis=0)#inputs
    y=np.concatenate((covid_features[:,-1],normal_features[:,-1],pneumonia_features[:,-1]), axis=0)#target labels
    # =============================================================================
    # normalization
    # =============================================================================
    min_max_scaler=MinMaxScaler()
    scaler = min_max_scaler.fit(X)
    X_test = scaler.transform(X_test)
    X = scaler.transform(X)

    # =============================================================================
    # feature reduction (K-PCA)
    # =============================================================================
    transformer = KernelPCA(n_components=64, kernel='linear')
    pca = transformer.fit(X)

    X_new_pca = pca.transform(X_test)
    #print(X_new_pca.shape)
    # min_max_scaler=MinMaxScaler()
    # X = X.reshape(-1,1)
    #X_new_pca = min_max_scaler.fit_transform(X_new_pca) 
    # print('X shape after normalization: ',X.shape)
    # X = X[:,:1].T
    # transformer = KernelPCA(n_components=64, kernel='linear')
    # X = X[:,:1].T
    # X = transformer.fit_transform(X)
    # print("X shape after KernelPCA: ",X.shape)
    trained_model = tf.keras.models.load_model('saved_model/my_model')

    Y_Score=trained_model.predict(X_new_pca)
    y_pred = np.argmax(Y_Score, axis=1)
    return Y_Score