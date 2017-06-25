# **Behavioral Cloning the car control** 

## Project overview
This project is all about capturing the human subcongnitive driving skills and reproduce them using convolutional neural network model. 
We use the simlator developed by the Udacity. Drive the car around the track and collect images from center, left and right camera along with 
the corresponding steering angles. Create the model and train it with the collected data and store the weights after training. The trained 
convolution neural network will be used to drive the car around the track in autonomus mode.

---

**Behavrioal Cloning Project**

---

My project includes the following files:
* model.py containing the script to create and train the model
* model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* model.json containing a trained convolution neural network
* README.md summarizing the results

### Model Architecture and Training Strategy

#### 1. Nvida model has been used with minor modifications.

The details of the layer is ashown below

![Alt text](/Model_layers.jpg?raw=true "Model Visualization")

 Nvidia model worked great. Minor tweeks are performed to develop a better model.
 
1. Added pooling layers
2. Added droupout layer to avoid overfitting of the model. Layers can be seen in run_save_model() function in the utilityFile.py
3. Image resizing

#### 2. Appropriate training data

(I have used the training dataset provided by Udacity)

Training data can be chosen to keep the vehicle driving on the road. Data should consist of center lane driving along with recovering data from the left and tight sides of the road. After training the model by using the data drom center camera car failed to recover during long turns. The training data size was increased by using the left and right camera images. correction factor 0.15 was used to get new corrected angles. Also the images were flipped and angles were multiplied by -1.0 as part of data augumentation. Normalization layer is included using lambda layer. The regular frame sized image used took lot of time in the process of training. So to reduce the calulation in pipeline i image cropping and resizing is done so that only important information is used in training.  Normalization, image cropping and image resizing is also added in the drive.py to match the trained model data.

##### Raw data        
  The original data captured from the simulator is of shape 160,320,3 and size 153600
 
###### Image from Center Camera                                
![Alt text](/originalcenter.png?raw=true "Original center image")   

###### Image from Left Camera 
![Alt text](/originalleft.png?raw=true "Original left image") 

###### Image from Right Camera
![Alt text](/originalright.png?raw=true "Original right image")

###### Cropped and Resized data
  The sky and bonnet section in the image has been cropped and resized to 64,64,3 as shown below to improve the processing speed of the pipeline.The preprocessed data is of shape 64,64,3 and size 12288

###### Image from Center Camera
![Alt text](/Resizedcenter.png?raw=true "Resized center image")

###### Image from Left Camera 
![Alt text](/Resizedleft.png?raw=true "Resized left image") 

###### Image from Right Camera
![Alt text](/Resizedright.png?raw=true "Resized right image")

The resized images are then agumented by flipping the image. this helps to generealize the model and the track is not memorized during training. This data is then normalized before feeding into the model during training.

#### 3. Process involved in Training 

After the collection process and preprocessing, the data was randomly shuffled and split into 80% training data and 20% validation data. 

Training data was used for training the model. The validation set helped to determine if the model was over or under fitting after every iteration. The model was trained and validated on different sets to ensure that the model was not overfitting. Adam optimizer has been used so that setting of the learning rate for training is not necessary.

#### 4. Testing the model
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```
