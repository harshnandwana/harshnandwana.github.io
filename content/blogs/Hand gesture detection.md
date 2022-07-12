---
title: "Hand Gesture detection"
date: 2022-07-07T03:04:04+05:30
summary: "3D Hand Posture Recognition from SmallUnlabeled Point Sets using techniques of Machine learning and Deep learning"
image: /images/summary5.jpg
social: false
---


## Introduction

This project covers use of Deep learning to evaluate Hand gestures captured by movement of sensors attached on them.

## Datasets

Dataset for this problem was povided by [ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/Motion+Capture+Hand+Postures)

To capture this data a Vicon motion capture camera system was used to record 12 users performing 5 hand postures with markers attached to a left-handed glove.

A rigid pattern of markers on the back of the glove was used to establish a local coordinate system for the hand, and 11 other markers were attached to the thumb and fingers of the glove. 3 markers were attached to the thumb with one above the thumbnail and the other two on the knuckles. 2 markers were attached to each finger with one above the fingernail and the other on the joint between the proximal and middle phalanx.

The 11 markers not part of the rigid pattern were unlabeled; their positions were not explicitly tracked. Consequently, there is no a priori correspondence between the markers of two given records. In addition, due to the resolution of the capture volume and self-occlusion due to the orientation and configuration of the hand and fingers, many records have missing markers. Extraneous markers were also possible due to artifacts in the Vicon software's marker reconstruction/recording process and other objects in the capture volume. As a result, the number of visible markers in a record varied considerably.

The data presented here is already partially preprocessed. First, all markers were transformed to the local coordinate system of the record containing them. Second, each transformed marker with a norm greater than 200 millimeters was pruned. Finally, any record that contained fewer than 3 markers was removed. The processed data has at most 12 markers per record and at least 3. For more information, see 'Attribute Information'.

Due to the manner in which data was captured, it is likely that for a given record and user there exists a near duplicate record originating from the same user. We recommend therefore to evaluate classification algorithms on a leave-one-user-out basis wherein each user is iteratively left out from training and used as a test set. One then tests the generalization of the algorithm to new users. A 'User' attribute is provided to accomodate this strategy.

This dataset may be used for a variety of tasks, the most obvious of which is posture recognition via classification. One may also attempt user identification. Alternatively, one may perform clustering (constrained or unconstrained) to discover marker distributions either as an attempt to predict marker identities or obtain statistical descriptions/visualizations of the postures.

## Data Visualization

Data here is provided in a csv file named Postures.csv which consists of 35 columns containing 
ClassID, User and 3 readings per sensor for 11 sensors. Here Class is a categorical feature that comprises of 5 unique values each represensting a specific postures with \
1=Fist(with thumb out), \
2=Stop(hand flat), \
3=Point1(point with pointer finger), \
4=Point2(point with pointer and middle fingers), \
5=Grab(fingers curled as if to grab). 


{{< figure src="/posture/1.png#center" width="400" >}}


``` python
def plot_final(Data,return_image=False):
    image1=cv2.imread(Data)
    image = aug.augment_image(image1)
    image=image[:,:,1]
    image[image <0.2]=0.5
    image = image / 255
    predicted  = unet.predict(image[np.newaxis,:,:])
    predicted[predicted <0.25]=0
    img = predicted[0,:,:,0]
    mean,std=cv2.meanStdDev(img)
    pixels = cv2.countNonZero(img)
    image_area = img.shape[0] * img.shape[1]
    area_ratio = (pixels / image_area) * 100
    img = img*255
    img[img<1]=1
    img[img>100]=255
    M= cv2.moments(img)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if return_image:
      return img,area_ratio,std,(cX,cY)
    else:
      return area_ratio,std,(cX,cY)
```