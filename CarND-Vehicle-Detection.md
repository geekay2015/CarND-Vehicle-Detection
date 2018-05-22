
# Vehicle Detection and Tracking
  
The project was about using Computer Vision and Machine Learning techniques for Vehicle Detection with Python + OpenCV to draw boxes on the cars in the road, first on images and afterwards on a video stream to create the final output video showed below.

## The goals / steps of this project are the following:
1. Data Exploration and visualization.
2. Perform a HOG feature extraction on a labeled training set of images.
3. Train classifier to detect cars using Linear SVM classifier with rbf kernel.
4. Normalize the features and randomize a selection for training and testing.
5. Implement a sliding-window technique and the trained classifier to search for vehicles in images and detect hot boxes
6. Estimate a bounding box for vehicles detected
7. Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4)


## Rubric Files
1. `CarND-Vehicle-Detection.ipynb` 
    * Contains the complete for code for the advance lane line detection pipelines
    * The file can be seen at [MyProjectCode](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/CarND-Vehicle-Detection.ipynb)
    
2. `CarND-Vehicle-Detection.md`
    * conatins all the rubric points and how I addressed each one with examples.
    * The file can be seen at [MyProjectWritup](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/CarND-Vehicle-Detection.md)
    
3.  `output_files`
    * a directory containg the output files from AdvancedLaneFinding.ipynb jupyter notebook.
    * The output files can be seen [MyOutputFIles](https://github.com/geekay2015/CarND-Vehicle-Detection/tree/master/output_images)
    
4. `project_video_output.mp4`
    * Complete Video Implementation from my pipeline 
    * Here's a link to my project video output
   
    [![Vehicle detection and tracking](http://img.youtube.com/vi/YrW0n-tNQjY/0.jpg)](https://www.youtube.com/watch?v=YrW0n-tNQjY)
   
   

## Data Exploration and Visualization
First I started by reading in all the vehicle and non-vehicle images provided. 
Import the labelled datasets which has vehicle and not vehicle images in `training_images` directory.

Here are links to the Labeled data for 
* [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and 
* [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) 

Examples to train your classifier.  These example images come from a combination of the 
* [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html)
* [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/)

The training dataset has 
* 8792 cars and 
* 8968 Non Cars

Here are some of the example from Vehicles class
![png](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/output_images/CarND-Vehicle-Detection_4_1.png)

Here are some of the example from Non-Vehicles class
![png](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/output_images/CarND-Vehicle-Detection_4_3.png)

## HOG feature extraction on a labeled training set of images

**HOG (Histogram of gradient descents)** is a powerful computer vision technique to identify the shape of an object using the direction of gradient along its edges. I have implemented in using `get_hog_features` function. 

The key parameters are 
* `orientations` - Orientations is the number of gradient directions. 
* `pixels_per_cell` - The pixels_per_cell parameter specifies the cell size over which each gradient histogram is computed. 
* `cells_per_block` - The cells_per_block parameter specifies the local area over which the histogram counts in a given cell will be normalized.

My feature vector consist of 128 components which I extract from grayscaled images, since grayscaled images contains all sctructure information. I beleive that it is better to detect cars by only structure information and avoid color information because cars may have big variety of coloring. Small amount of featues help to make the classifier faster while loosing a little amount of accuracy. 

My parameters of feature extraction are


```python
# parameters of feature extraction

color_space = 'GRAY' # Can be GRAY, RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 1 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off

```
![png](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/output_images/CarND-Vehicle-Detection_8_0.png)

![png](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/output_images/CarND-Vehicle-Detection_8_1.png)

![png](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/output_images/CarND-Vehicle-Detection_8_2.png)

![png](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/output_images/CarND-Vehicle-Detection_9_0.png)

![png](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/output_images/CarND-Vehicle-Detection_9_1.png)

![png](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/output_images/CarND-Vehicle-Detection_9_2.png)


## Training classifier to detect cars using Linear SVM 
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of SVMs are:
* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory   efficient.
* Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of SVMs are:
* If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

```
result dataset: 8792 cars / 8968 not cars
features extraction time:  17.79
```

## Feature Normalization
I used [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) for feature normalization along resulting dataset. 

`StandardScaler` Standardize features by removing the mean and scaling to unit variance. Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using the transform method.

Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual feature do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

For instance many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines) assume that all features are centered around 0 and have variance in the same order. If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

```
Using: 8 orientations 16 pixels per cell and 1 cells per block
Feature vector length: 128
```

## Randomize a selection for training and testing
I Normalize my features and randomized the selection for training and testing. My feature vector length of 128 components.
```
Using: 8 orientations 16 pixels per cell and 1 cells per block
Feature vector length: 128
```

## Test the Classifier Accuracy

This turned out be a really importaint step as it adds about ~4% accuracy for my classifier. Length of my feature vector consist of 128 components and resulting accuracy is 98%.

```
Test Accuracy of SVC =  0.9887
```

##  Sliding Window Implementation to Search and Classify

For searching cars in an input image I use sliding window technique. It means that I iterate over image area that could contain cars with approximately car sized box and try to classify whether box contain car or not. As cars may be of different sizes due to distance from a camera we need a several amount of box sizes for near and far cars. I use 3 square sliding window sizes of 128, 96 and 80 pixels side size. While iterating I use 50% window overlapping in horizontal and vertical directions. One of sliding window drawn in blue on each image while rest of the lattice are drawn in black. For computational economy and additional robustness areas of sliding windows don't conver whole image but places where cars appearance is more probable.

So, my goal here is to write a function that takes in an image, start and stop positions in both x and y (imagine a bounding box for the entire search region), window size (x and y dimensions), and overlap fraction (also for both x and y). Your function should return a list of bounding boxes for the search windows, which will then be passed to draw draw_boxes() function.

![png](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/output_images/CarND-Vehicle-Detection_21_0.png)


## Estimate a bounding box for vehicles detected
So far I trained the classifier, then ran my sliding window search, extract features, and predict whether each window contains a car or not.

I used the below functions to search over all the windows defined by your `slide_windows()`, `extract features` at each window position, and predict with my classifier on each set of features.
* `single_img_features()` and 
* `search_windows()`

Number of hot boxes crowded at real cars positions along with less number of hot boxes crowded in false positives of classifier. We need to estimate real cars positions and sizes based on this information. 

Below is the algorithm:-
* Apply a sliding windows to images and finds hot windows. Also return image with all hot boxes are drawn
* Computes heat map of hot windows. Puts all specified hot windows on top of each other, so every pixel of returned image will contain how many hot windows covers this pixel
* Average Hot boxes algorithm.
    * Get the number of joined boxed
    * Use joined boxes information to compute this average box representation as hot box. This box has average center of all boxes and have size of 2 standard deviation by x and y coordinates of its points
    * Check wether specified box is close enough for joining to be close need to overlap by 30% of area of this box or the average box
    * Join in all boxes from list of given boxes, remove joined boxes from input list of boxes
    * Compute average boxes from specified hot boxes and returns average boxes with equals or higher strength  

```
    /Users/gangadharkadam/anaconda/envs/carnd-term1/lib/python3.5/site-packages/skimage/feature/_hog.py:119: skimage_deprecation: Default value of `block_norm`==`L1` is deprecated and will be changed to `L2-Hys` in v0.15
      'be changed to `L2-Hys` in v0.15', skimage_deprecation)
```

![png](https://github.com/geekay2015/CarND-Vehicle-Detection/blob/master/output_images/CarND-Vehicle-Detection_24_1.png)


## Video Implementation
In the video I used information from multiple frames to make average boxes more robust and filter false positives. I accumulate all hot boxes from last several frames and used them for calculating average boxes.

```
    [MoviePy] >>>> Building video test_video_result.mp4
    [MoviePy] Writing audio in test_video_resultTEMP_MPY_wvf_snd.mp3


    100%|██████████| 34/34 [00:00<00:00, 88.50it/s]

    [MoviePy] Done.
    [MoviePy] Writing video test_video_result.mp4


    
     97%|█████████▋| 38/39 [00:10<00:00,  3.83it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_video_result.mp4 
    
    CPU times: user 9.03 s, sys: 793 ms, total: 9.82 s
    Wall time: 11.9 s
```

```
    [MoviePy] >>>> Building video project_video_result.mp4
    [MoviePy] Writing audio in project_video_resultTEMP_MPY_wvf_snd.mp3


    100%|██████████| 1112/1112 [00:01<00:00, 634.60it/s]

    [MoviePy] Done.
    [MoviePy] Writing video project_video_result.mp4


    
    100%|█████████▉| 1260/1261 [05:43<00:00,  3.84it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: project_video_result.mp4 
    
    CPU times: user 5min 6s, sys: 26.8 s, total: 5min 33s
    Wall time: 5min 46s
```

Here's a Link to my video result

[![Vehicle detection and tracking](http://img.youtube.com/vi/YrW0n-tNQjY/0.jpg)](https://www.youtube.com/watch?v=YrW0n-tNQjY)

## Discussion

Detecting cars with SVM in sliding windows is interesting method but it has a number of disadvantages. While trying to make my classifier more quick I faced with problem that it triggers not only on cars but on other parts of an image that is far from car look like. So it doesn't generalizes well and produces lot of false positives in some situations. Also sliding windows slowes computation as it requires many classifier tries per image. Again for computational reduction not whole area of input image is scanned. So when road has another placement in the image like in strong curved turns or camera movements sliding windows may fail to detect cars.

I think this is interesting approach for starting in this field. But it is not ready for production use. I think convolutional neural network approach may show more robustness and speed. As it could be easily accelerated via GPU. Also it may let to locate cars in just one try. For example we may ask CNN to calculate number of cars in the image. And by activated neurons locate positions of the cars. In that case SVM approach may help to generate additional samples for CNN training.
