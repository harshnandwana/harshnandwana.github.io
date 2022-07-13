---
title: "Brain MRI Segmentation"
date: 2022-07-07T22:41:10+05:30
darft: true
summary: "Solving task of Image segmentation to detect Tumor and attaining best Dice score ever"
image: /summary/8.jpg
social: false
---

## Introduction

In this blog, we will try to solve a famously discussed task of Brain MRI segmentation. Where our task will be to take brain MR images as input and utilize them with deep learning for automatic brain segmentation matured to a level that achieves performance near to a skilled radiologist and then predicts whether a person is identified with a tumor or not.

If there is a tumor detected then we need to provide as much information about the tumor as possible so that this information can be used by doctors to provide better treatment to the patient so and at last we will also try to detect whether a person will survive or not.

In this task, we utilize knowledge of both worlds from deep learning to radiology on this dataset provided by The Cancer Genome Atlas (TCGA) and The Cancer Imaging Archive (TCIA) of 110 different patients and try to generate imaging biomarkers that could provide us with information about the tumor.

This can be classified as a task for supervised learning where we are provided with masks for each image and even our task is to create an image with masks

Lower-grade gliomas are a group of WHO grade II and grade III brain tumors including well-differentiated and anaplastic astrocytomas, oligodendrogliomas, and oligoastrocytomas

## Dataset

Dataset for this problem was previously made publicly available by TCGA. Data provided is of image type with .tif extension and is present in folder with extension ”lgg-mri-segmentation/kaggle_3m/” . This extension here represents all train MR image files whereas for masks extension ends with “mask”. Apart from these, we are also given a data.csv file that consists of all genome information about each patient.

In the Medical Engineering domain, we generally find less data where a disease is present so generally datasets here are highly imbalanced. But in this case, we find it to be almost balanced.
<p align="center">
  <img src="/images/brain/1.png" alt="hello!" title="adam solomon's hello" width="400"/>
</p>

As there are around 3800 images of only 110 patients. This is as we have multiple scan ranging from 20 to 88 per patient.

<p align="center">
  <img src="/images/brain/2.png" alt="hello!" title="adam solomon's hello" width="700"/>
</p>

To make this task simpler this dataset was manually labeled by creator so as we can classify it into supervised learning problem.

This Dataset can be downloaded from [here](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) 

## Method

As our primary objective is to identify more and more features of the tumor so that information can be used by doctors to cure them. some important points that we believe will be beneficial for doctors will be

1.Area of tumor
2.coordinates of tumor
3.shape/spread of the tumor.

Gathering all this information we can utilize them to predict almost all visible details about a tumor.

To obtain these data

1. Preprocess data provided and convert it model feedable format
2. Decide best suitable metric and losses
3. Build a best suitable model architecture
4. Tune model performance
5. Predict tumor mask
6. Analyze the area of mask obtained in which values of pixels are non zero
7. Calculate coordinates of the centroid of non-zero pixels
8. Obtain std deviation
9. Display results
10. use these results to predict death of a patient 

## Data preprocessing

Data preprocessing here is most crucial step as here we do most of our preprocessing and feature engineering stuff. That turns out to be one of major feature in our case study solution

First lets have a look at all MR images present for a single patient "TCGA_CS_4941". Here red circle shows the area where you can identify a tumor
<p align="center">
  <img src="/images/brain/3.png" alt="hello!" title="adam solomon's hello" width="700"/>
</p>
Now as we can see that there are significant numbers of images with tumors. but as we are not a trained radiologist or doctor so it turned out that we need to develop some masked image using already test given masks
<p align="center">
  <img src="/images/brain/4.png" alt="hello!" title="adam solomon's hello" width="400"/>
</p>
From above images we can observe that not all colors in a image are equally useful. it turns out that whenever there is a tumor it gets highlighted with green color thus we can say that whenever there is high intensity green color there may be a tumor, also images here are not too sharp to get high intensity of each color therefore we also need a image sharpener, and we don't need to perform image augmentation as these are image outputs from a standard medical machine. On basis of this observation we can create a custom data loader class for image preprocessing and data loading combined that should work on this flow chart
<p align="center">
  <img src="/images/brain/5.png" alt="hello!" title="adam solomon's hello" width="400"/>
</p>
Code:

``` python
aug = iaa.Sharpen(alpha=(1.0), lightness=(1.5))

def adjust_data(img,mask):
    #img = img[:,:,1]
    #print(img.shape)
    img[img <0.2]=0.5
    img = img / 255
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img,mask)

class Dataset:
    # we will be modifying this CLASSES according to your data/problems
    
    # the parameters needs to changed based on your requirements
    # here we are collecting the file_names because in our dataset, both our images and maks will have same file name
    # ex: fil_name.jpg   file_name.mask.jpg
    def __init__(self, dataframe):
        
        self.ids = dataframe['Patient']
        # the paths of images
        self.images_fps   = dataframe['img']
        # the paths of segmentation images
        self.masks_fps    = dataframe['mask']
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_UNCHANGED)
        image = aug.augment_image(image)
        image=image[:,:,1]
        image= np.reshape(image, (256,256,1))
        
        image = image.astype(np.float32)
        
        mask  = cv2.imread(self.masks_fps[i], cv2.IMREAD_UNCHANGED)
        mask = np.reshape(mask, (256,256,1))
        image_mask = mask
        image_mask = image_mask.astype(np.float32)
        
        image,image_mask= adjust_data(image, image_mask)
        return (image,image_mask)
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(tf.keras.utils.Sequence):    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        batch = [np.stack(samples) for samples in zip(*data)]
        
        return tuple(batch)
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
```

Here in class Dataset we just need to pass a pandas data-frame with image path and mask path along with patient name and it will return a tuple that contains image and mask .

This tuple is then being passed in Dataloader where based on batch size provided it is being transformed into a model loadable data set.
code:

<p align="center">
  <img src="/images/brain/6.png" alt="hello!" title="adam solomon's hello" width="400"/>
</p>

Here you can see i manually marked area with tumor for red color and also you can observe that its fairly easy to visualize these area as these are already marked with high intensity green color.

## Metrics and Losses

As these are tasks for image segmentation. therefore there evaluation metrics are non trivial to solve. In this we need a pixel wise comparison between both actual mask and predicted mask.

Therefore there are 2 proposed metrics for semantic segmentation tasks

### IOU (Jaccard Index)

Jaccard Index, is one of the most commonly used metrics in semantic segmentation as IoU  can be defined as area of overlap between predicted segmentation and the ground truth divided by the area of union between the predicted segmentation and the ground truth. IOU is defined in range(0-1). Here 0 is defined as no area is being overlapped whereas 1 is defined as no noise and entire defined area is being overlapped.

<p align="center">
  <img src="/images/brain/7.png" alt="hello!" title="adam solomon's hello" width="200"/>
</p>
https://en.wikipedia.org/wiki/Jaccard_index

### Dice Score

Dice score is a useful score that we will use in our case study for evaluation as this metric was first used in paper and till then it is being used to compare your model against others

Dice Coefficient = 2 * the Area of Overlap divided by the total number of pixels in both images.
<p align="center">
  <img src="/images/brain/8.png" alt="hello!" title="adam solomon's hello" width="200"/>
</p>

losses and metrics can be obtained in Keras using

code:

``` python
import tensorflow.keras.backend as K                                        

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + 100) / (K.sum(y_truef) + K.sum(y_predf) + 100))

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + 100) / (sum_ - intersection + 100)
    return jac      
```

## Model Selection

After going through various model proposed for bio medical image segmentation model proposed we came to conclusion to use Unet and versions of Unets along with transfer learning. Use of transfer learning will help us reduce training time significantly and obtain better accuracy as using Unet with resnet50 provides an architecture where Resnet 50 acts as a backbone that helps to detect features in images and is pretrained with image net datasets.

### Unet++

This is a Unet architecture with lots of skip connections these skip connections help to obtain particle size features from a image
<p align="center">
  <img src="/images/brain/9.png" alt="hello!" title="adam solomon's hello" width="400"/>
</p>
https://arxiv.org/abs/1807.10165

This architecture can be implemented in keras as 
<p align="center">
  <img src="/images/brain/10.png" alt="hello!" title="adam solomon's hello" width="700"/>
</p>
This predicts mask with Dice score of 0.9 which is a good score and its predicted image can be viewed as
<p align="center">
  <img src="/images/brain/11.png" alt="hello!" title="adam solomon's hello" width="700"/>
</p>

### Unet with Resnet as Backbone

This is an architecture with Resnet encoders as backbone and weight of these encoders are freezed.
<p align="center">
  <img src="/images/brain/12.png" alt="hello!" title="adam solomon's hello" width="400"/>
</p>
https://arxiv.org/abs/2204.12084

This shows an exceptional Dice score of 0.946
<p align="center">
  <img src="/images/brain/13.png" alt="hello!" title="adam solomon's hello" width="700"/>
</p>

### Comparison

Here we can see results of each model with DICE and IOU metric and we can also conclude unetxresnet is architecture that fits our needs.
<p align="center">
  <img src="/images/brain/14.png" alt="hello!" title="adam solomon's hello" width="400"/>
</p>

## Feature Calculations

Now lets calculate some important features that will be helpful for doctors to analyze condition of a patient.

Code:
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

This function returns area, standard deviation and coordinates
<p align="center">
  <img src="/images/brain/15.png" alt="hello!" title="adam solomon's hello" width="700"/>
</p>

## Predict Death

Death01 is a feature present in “lgg-mri-segmentation/kaggle_3m/data.csv” which tells us whether a patient is going to die or not. But there are lots of missing values present in this sheet that needs to be filled up.
<p align="center">
  <img src="/images/brain/16.png" alt="hello!" title="adam solomon's hello"/>
</p>
To fill all these unknown values we use an imputer. Here we decide to choose KNNImputer from scikitlearn with n-neighbors=4 and then round it off so as to obtain integer values from the float.

Join both these data frames based on patient ID as key and method as inner join to create another

Now these features can be utilized with provided Data.csv file to predict death01 features as y and it seems that we are able to classify all our points with 100% accuracy
<p align="center">
  <img src="/images/brain/17.png" alt="hello!" title="adam solomon's hello"/>
</p>

## Deployment

Now the major task is to feed this with any image of .tif format and it should be capable of creating a mask for that image and generating above discussed features. We will not be taking care of Data.csv here. Here we will just generate masks and important features.

For this we will be using stream lit and further code can be downloaded from [here](https://github.com/harshnandwana/Brain-mri-segmentation/tree/main/Deploy).

{{< youtube 9hwXkstQYZA >}}

## Conclusion

This [case study](https://www.analyticsvidhya.com/blog/2022/01/a-guide-to-understand-machine-learning-pipeline-with-case-study/) discusses various approaches that can be used in process of solving a conventional 2D image segmentation using Keras and Tensorflow. It also discusses what should be the appropriate loss functions and evaluation metrics and how we can just utilize 1 channel from RGB based on EDA to obtain an image with lower dimensions which will help us reduce time and increase performance.  We also discussed which features will make a significant impact on doctors and can be extracted from images. On comparing this with other solutions available we found that this approach provides us with the best Dice score ever. Below are some of the key takeaways:

* How Deep-learning and transfer learning can be utilized to solve the task of Biomedical image segmentation
* What should be the best loss function and evaluation metrics for our task
* Generating features from images and utilizing them to predict the death of a patient with 100% accuracy
* Use Streamlit to deploy our model for simpler use

You can find the whole code here:  https://github.com/harshnandwana/Brain-mri-segmentation