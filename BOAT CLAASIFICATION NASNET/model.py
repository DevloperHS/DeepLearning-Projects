#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rasterio,os,sys
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from rasterio.plot import show,show_hist
from rasterio.windows import Window
from rasterio.plot import reshape_as_raster, reshape_as_image
from sklearn.utils import shuffle
from sklearn import preprocessing,metrics,pipeline
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn import externals,feature_selection 
from sklearn import naive_bayes,linear_model,ensemble,tree,svm 
from sklearn.model_selection import GridSearchCV
#from IPython.display import Image
#import matplotlib as mlp
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import earthpy.clip as ec
import rasterio as rio
from glob import glob
from shapely.geometry import mapping
from sklearn.cluster import KMeans
import pickle


# In[2]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=Warning)


# In[3]:


def load_landsat_scenes(directory,scenes):
    os.chdir(os.path.join(et.io.HOME, directory))
    return glob(scenes)


# In[4]:


#read shapefile and reproject to match raster
def read_shapefile(shapefile):
    shapefile = gpd.read_file(shapefile)
    return shapefile.to_crs({'init': 'epsg:32638'})


# In[5]:


def stack_landsat(landsat_band_paths, img_dir):
    # create a products directory within the data dir which won't be uploaded to Github
    #img_dir = 'product/'
    #'../UNHCR-Challege/Banadir/LC08_L1TP_163058_20150118_20170414_01_T1/product/'

    # check to see if the dir it exists, if not, create it
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # filepath for image we're writing out
    img_fp = img_dir + 'landsat_bands.tif'

    # Read metadata of first file and assume all other bands are the same
    with rasterio.open(landsat_band_paths[0]) as src0:
        meta = src0.meta

    # Update metadata to reflect the number of layers
    meta.update(count = len(landsat_band_paths))

    # Read each layer and write it to stack
    with rasterio.open(img_fp, 'w', **meta) as dst:
        for id, layer in enumerate(landsat_band_paths, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))
    return img_fp
    


# In[6]:


def load_stacked_image(img_fp):
    full_dataset = rasterio.open(img_fp)
    img_rows, img_cols = full_dataset.shape
    img_bands = full_dataset.count
    return full_dataset


# In[7]:


def features_from_shapefile(shapefile):
    # this generates a list of shapely geometries
    geoms = shapefile.geometry.values 
    #print(geoms)
    # let's grab a single shapely geometry to check
    geometry = geoms[1] 
    # transform to GeoJSON format

    feature = [mapping(geometry)]
    return feature


# In[8]:


def mask_image(full_dataset,features):
    #Create extract the raster values values within the polygon 
    out_image, out_transform = mask(full_dataset, features, crop=True)
    return out_image,out_transform


# In[9]:


def pre_image_classification(img_fp,feature,shapefile):
    # build training data for classification
    X = np.array([], dtype=np.int8).reshape(0,8) # pixels for training
    y = np.array([], dtype=np.string_) # labels for training
    geoms = shapefile.geometry.values 
    with rasterio.open(img_fp) as src:
        band_count = src.count
        for index, geom in enumerate(geoms):
            feature = [mapping(geom)]

            # the mask function returns an array of the raster pixels within this feature
            out_image, out_transform = mask(src, feature, crop=True) 
            # eliminate all the pixels with 0 values for all 8 bands - AKA not actually part of the shapefile
            out_image_trimmed = out_image[:,~np.all(out_image == 0, axis=0)]
            # eliminate all the pixels with 255 values for all 8 bands - AKA not actually part of the shapefile
            out_image_trimmed = out_image_trimmed[:,~np.all(out_image_trimmed == 255, axis=0)]
            # reshape the array to [pixel count, bands]
            out_image_reshaped = out_image_trimmed.reshape(-1, band_count)
            # append the labels to the y array
            y = np.append(y,[shapefile["name"][index]] * out_image_reshaped.shape[0]) 
            # stack the pizels onto the pixel array
            X = np.vstack((X,out_image_reshaped)) 
    return y,X


# In[10]:


# convert the class labels into indicies 
def str_class_to_int(class_array):
    class_array[class_array == 'buildings'] = 0
    class_array[class_array == 'landuse'] = 1
    class_array[class_array == 'natural'] = 1
    return(class_array.astype(int))


# In[11]:


# proceed to classify the clipped image: but limited 
# By memory so we take a slice of the image
# may need to reduce this image size if your kernel crashes,
#takes a lot of memory
def clip_image(img_fp):
    with rasterio.open(img_fp) as src:
    
        img = src.read()[:, 400:1500, 1500:3000]

        # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
        reshaped_img = reshape_as_image(img)
        
    return img,reshaped_img
    


# In[12]:


def normalize(X):
    X1 = X.astype(np.float)
    scaler = preprocessing.StandardScaler()
    X1 = scaler.fit_transform(X1)
    return X1


# In[13]:


# stratified cross validation to cater for class inbalance between hams and spam
def stratified_cv(X, y, clf_class, shuffle=True, n_splits=10, **kwargs):
    X = normalize(X)
    stratified_k_fold = StratifiedKFold(n_splits=n_splits,  shuffle=shuffle)
    y_pred = y.copy()
    for ii, jj in stratified_k_fold.split(X, y):
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf= clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    return y_pred


# In[14]:


def color_stretch(image, index):
    colors = image[:, :, index].astype(np.float64)
    for b in range(colors.shape[2]):
        colors[:, :, b] = rasterio.plot.adjust_band(colors[:, :, b])
    return colors


# In[15]:


def plot_visualization(reshaped_img,predictions,cmap):
    fig, axs = plt.subplots(1,2,figsize=(10,7))

    img_stretched = color_stretch(reshaped_img, [4, 3, 2])
    axs[0].imshow(img_stretched)
    axs[0].set_title('RGB of The Bands')
    axs[1].imshow(class_prediction, cmap=cmap, interpolation='none')
    axs[1].set_title('Result of the Classification')
    #plt.show(block=False)
    fig.savefig('output/predictions_visualization.png')


# In[16]:


def ndvi_ndwi(img_fp):
    with rasterio.open(img_fp) as src:
        green_band = src.read(3)
        red_band = src.read(4)
        nir_band = src.read(8)
    
    ndwi = (green_band.astype(float) - nir_band.astype(float)) / (green_band.astype(float) + nir_band.astype(float))
    ndvi = (nir_band.astype(float) - red_band.astype(float)) / (red_band.astype(float) + nir_band.astype(float))
    return ndwi,ndvi


# In[17]:


def plot_ndvi_ndwi(img_fp,reshaped_img,predictions,cmap):
    #NDWI is similar to NDVI but for identifying water.
    #that are high on the NDWI ratio are generally classified
    #as water and areas high on NDVI are forest
    #clip ndwi and ndvi
    ndwi,ndvi = ndvi_ndwi(img_fp)
    ndwi = ndwi[400:1500, 1500:3000]
    ndvi = ndvi[400:1500, 1500:3000]
    fig, axs = plt.subplots(2,2,figsize=(15,7))

    img_stretched = color_stretch(reshaped_img, [3, 2, 1])
    axs[0,0].imshow(img_stretched)
    axs[0,0].set_title('RGB of The Bands')
    axs[0,1].imshow(class_prediction, cmap=cmap, interpolation='none')
    axs[0,1].set_title('The Classification')
    nwdi_plot = axs[1,0].imshow(ndwi, cmap="RdYlGn")
    axs[1,0].set_title("NDWI")
    fig.colorbar(nwdi_plot, ax=axs[1,0])

    ndvi_plot = axs[1,1].imshow(ndvi, cmap="RdYlGn")
    axs[1,1].set_title("NDVI")
    fig.colorbar(ndvi_plot, ax=axs[1,1])

    #plt.show(block=False)
    fig.savefig('output/ndvi_ndwi.png')


# In[18]:


def plot_confusion_matrix(y,X):
    # Confusion matrix to find out clearly how many TP, TN, FP and FN of each technique
    random_forest_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, ensemble.RandomForestClassifier))

    fig,ax = plt.subplots(figsize=(12, 7))
    plt.title('Confusion Matrix', fontsize = 14)

    sns.heatmap(random_forest_conf_matrix,  annot=True, fmt="d",ax=ax)
    #plt.show(block=False)
    fig.savefig('output/confusion_matrix.png')


# In[19]:


def computeKmeans(clusters,img,reshape_img):
    bands, rows, cols = img.shape


    kmeans_predictions = KMeans(n_clusters=clusters, random_state=0).fit(reshaped_img.reshape(-1, 8))

    kmeans_predictions_2d = kmeans_predictions.labels_.reshape(rows, cols)

    # Now show the classmap next to the image
    fig, axs = plt.subplots(1,2,figsize=(15,8))

    img_stretched = color_stretch(reshaped_img, [3, 2, 1])
    axs[0].imshow(img_stretched)
    axs[0].set_title('RGB of The Bands')
    axs[1].imshow(kmeans_predictions_2d)
    axs[1].set_title('Kmeans Clustering')
    #plt.show(block=False)
    fig.savefig('output/kmeansClustering.png')


# In[20]:


def color_map(predictions):
    # find the highest pixel value in the prediction image
    n = int(np.max(predictions))

    # next setup a colormap for our map
    colors = dict((
        (0, (139,69,19, 255)),   # Buildings
        (1, (34, 139, 34, 255)),      # Land use
        (2, (96, 19, 134, 255)),    # Nature
    
    ))

    # Put 0 - 255 as float 0 - 1
    for k in colors:
        v = colors[k]
        _v = [_v / 255.0 for _v in v]
        colors[k] = _v
    
    index_colors = [colors[key] if key in colors else 
                (255, 255, 255, 0) for key in range(0, n+1)]

    cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', n+1)

    return cmap





if __name__=='__main__':
    
    working_directory = os.getcwd()

    #change this path for custom input
    scenes_path ='input/LC08_L1TP_163058_20150118_20170414_01_T1/LC08_L1TP_163058_20150118_20170414_01_T1_B*.TIF'
    filename = 'input/somalia.shp'
    
    #scenes_path = sys.argv[1]
    #filename = sys.argv[2]

    landsat_band_paths = load_landsat_scenes(working_directory,scenes_path)
    
    img_fp=stack_landsat(landsat_band_paths,'output/')
    
    full_dataset = rasterio.open(img_fp)
    
    
    shapefile =read_shapefile(filename)
    
    features = features_from_shapefile(shapefile)
    
    #out_image,out_transform = mask_image(full_dataset,features)
    
    y,X =pre_image_classification(img_fp,features,shapefile)
    
    
    img,reshaped_img = clip_image(img_fp)
    
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    classifier.fit(X, y)
    class_prediction = classifier.predict(reshaped_img.reshape(-1, 8))

    # Reshape our classification map back into a 2D matrix so we can visualize it
    class_prediction = class_prediction.reshape(reshaped_img[:, :, 0].shape)
    class_prediction = str_class_to_int(class_prediction)
    #pickle.dump('src/classifier.pkl')
    
    cmap = color_map(class_prediction)

    plot_visualization(reshaped_img,class_prediction,cmap)
    
    plot_ndvi_ndwi(img_fp,reshaped_img,class_prediction,cmap)
    
    clusters = 5
    
    computeKmeans(clusters,img,reshaped_img)
    
    #Model accuracy by cross validation: got to test damn Naive Bayes
    print('Random Forest Classifier accuracy:      {:.3f}'.format(metrics.accuracy_score                                                     (y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
    print('Naive Bayes accuracy:           {:.3f}'.format(metrics.accuracy_score                                                     (y, stratified_cv(X, y, naive_bayes.GaussianNB))))


    plot_confusion_matrix(y,X)
    #plt.show()







