# Custom-Object-Detection-Using-Darkflow-Simplified
This is a very simplified repository for custom object detection .If you want to just apply custom object detection using YOLO this is the best that you can get .It requires no prior Knowledge. So gear up and enjoy.

# Steps 
  - Setting up for installing tensorflow-gpu (if your GPU supports CUDA)
  - Image gathering 
  - Image annotation
  - Setting up environment
  - Setting up darkflow
  - File editing in Darkflow folder
  - Training
  - Predicting 
# Setting up for installing tensorflow-gpu (skip if you already have)
    
   
   This step is the most time taking step as during this step anyone can face lot of errors.So I will try my best to give you steps so that    you will not encounter any error.
   Note -- Install all the package and dependencies version same as give below to avoid any error.
   
   - First of all check your GPU is compatible with CUDA or not ?
   
     [CUDA comapatible GPUs](https://developer.nvidia.com/cuda-gpus)
   - Then install anaconda cloud.  [from here](https://www.anaconda.com/distribution/) 
     Note - while installing anaconda check"the anaconda add on my PATH variable"
     ![alt text](https://github.com/Boltuzamaki/Custom-Object-Detection-Using-Darkflow-Simplified-/blob/master/images%20support%20file/anaconda.PNG)
   - Then install Cuda Toolkit 9 [from here](https://developer.nvidia.com/cuda-90-download-archive).
     Download Base installer
   - After then download Visual studio 19 [from here](https://visualstudio.microsoft.com/downloads/).Download community version.
    After installing it go to modify option and select "Desktop development with c++" and modify.You only need to download this 
    because it supports CUDA.
    ![alt text](https://github.com/Boltuzamaki/Custom-Object-Detection-Using-Darkflow-Simplified-/blob/master/images%20support%20file/2.PNG)
    
   - Download cuDNN version which is latest and supported by CUDA 9. [from here](https://developer.nvidia.com/rdp/cudnn-archive).
     and then extract it.
     It has three folders bin,include,lib we have to add this location to path variable.
     For this,
     go to This PC->Properties ->Advance system setting -> Environment Variables ->Path 
     Then add location of all the three folders of cuDNN.
     
     This will look something like this
     ![alt text](https://github.com/Boltuzamaki/Custom-Object-Detection-Using-Darkflow-Simplified-/blob/master/images%20support%20file/3a.png?raw=true)
     
   - At last install tensorflow-gpu. (1.12.0 version is compatible with CUDA 9)
        **pip install --ignore-installed --upgrade tensorflow-gpu==1.12.0**  (And first unistall tensorflow to remove tensorflow CPU using **pip unistall tensorflow**)
        
# Image Gathering 

Now its time to gather a sufficient amount of image for training .

### Methods 
- If you want to train for simple looking image like an apple then just grap a cellphone and strat taking pictures of that object with different orientation and different backgrounds and diffrent angles try to make pictures much more diverse .
- Other method you can download images from internet.You can use various api for this on which I would not go in much more detail.

Note -- Try to take picture in a medium or low resolution to make training faster.

# Image annotation 
For image annotation there are many softwares but I prefer LabelImg which you can download from 
[from here](https://tzutalin.github.io/labelImg/).Download latest version.
--> If you know about YOLO then you know that for training we requires two types of file . One is image file and other is .xml file .XML file stores the co-ordinate and label of image present in the corresponding image.

Now open LabelImg and start labelling your image and save all XML files in a different folder.
You can see [THIS VIDEO ](https://www.youtube.com/watch?v=p0nR2YsCY_U) for how to use LabelImg?

# Setting up environment

Now as tensorflow gpu is install we need to install cv2 , PIL .So type the following in anaconda terminal
```sh
pip install cv2
```
```sh
pip install PIL
```
If you will face any error of library not found later then just install that library 

# Setting up darkflow

Now the main part comes to run YOLO easily.
**DarkFlow** is a network builder adapted from Darknet. It allows building TensorFlow networks from cfg. files and loading pre trained weights. .cfg file are simply summary of the YOLO model's layers .
So to setting up darkflow first clone and download the repository of darkflow [from here](https://github.com/thtrieu/darkflow) 

Now first open the folder of darkflow at first it will look like this
![alt text](https://github.com/Boltuzamaki/Custom-Object-Detection-Using-Darkflow-Simplified-/blob/master/images%20support%20file/im2.PNG?raw=true)

Next we have to build the darkflow for this open anaconda prompt then type 
```sh
cd "location_of_darkflow_folder"
```
```sh
python3 setup.py build_ext --inplace
```
```sh
pip install .
```

# Folder setup in Darkflow folder

Further we have to add some folders in it.And also we have to download pretrained weight so that we can use transfer learning and our model will train fast .
Downlaod YOLO weights  -- [from here](https://pjreddie.com/darknet/yolo/) 
As we can see below that there are many models and corresponding .cfg file.You may use any of these according to your requirement . I prefer "YOLOv2 608x608".

![alt text](https://github.com/Boltuzamaki/Custom-Object-Detection-Using-Darkflow-Simplified-/blob/master/images%20support%20file/yolo.PNG?raw=true)

Now time to create some folder in darkflow folder.
The structure of folder is look like this 
- bin
  - yolo.weights (It is the weight file that you downloaded above)
- Image (This folder contains your training images)
  - img1
  - img1
  - .....
- xml (This folder contains xml file of images that is created during labelling)
  - img1.xml
  - img2.xml
  - .....

- ckpt (This folder is created to store training checkpoints  )
   - cfg(folder leave this empty)

Note -- Take care that the number of images in Image folder and number of .xml files in xml folder must be same .If not then it will cause problem in running code .

Now your darflow folder should look like this 
![alt text](https://github.com/Boltuzamaki/Custom-Object-Detection-Using-Darkflow-Simplified-/blob/master/images%20support%20file/im3.PNG?raw=true)

Now setup part is done .Time for fun :)

# Training
- Before training open **cfg** folder and here paste the .cfg file that you have downloaded from YOLO site.(Take care that you paste the same corresponding cfg file as your .weights). Now time to edit cfg file as your custom datatset  .
In cfg file search classes and change the value of this according to the number of classes present in your custom training dataset.
- Second open notepad and type of the classes if your custom dataset in it and save it as labels.txt . It should look like the figure given below
![alt text](https://github.com/Boltuzamaki/Custom-Object-Detection-Using-Darkflow-Simplified-/blob/master/images%20support%20file/label.PNG?raw=true)

Time for training -- 

Now open Anaconda prompt and type spyder
Then open the code training.py from this repo
And run it 
Training will start in 30-60 seconds depending on your GPU 

        
        
     
     
     
    
    
    
    
      
     
   
 
     
     
   
   


  
