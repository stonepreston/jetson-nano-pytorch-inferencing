# Nano Setup

On your development machine, open a terminal window and ssh into nano. Enter your password when prompted.

    $ ssh your_nano_username@your_nano_ip

You will now be connected to your nano inside the terminal window. Create a project directory for your pytorch project in your home directory.

    $ cd 
    $ mkdir your_project_folder

## Python Virtual Environment Setup

We need to setup python on the nano. Using virtual environments is best practice to help with dependancy management. In this section, we will install a python package manager (pip) and setup a virtual environment using venv.

### Install and upgrade pip

    $ python3 -m pip install --user --upgrade pip

You can check whether it installed successfully using the command below:

    $ python3 -m pip --version
    
### Venv setup
 
 Install venv
 
    $ sudo apt-get install python3-venv
    
Now create a new virtual environment called env inside your project folder. Note that we use the -m flag (module) to use the venv python module.

    $ cd your_project_folder
    $ python3 -m venv env
    
 Activate the virtual environment
 
     $ source env/bin/activate
     
## Installing PyTorch

Installation steps are found [here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano-with-new-torch2trt-converter/) and are listed below.

### Install PyTorch for the Jetson Nano (Python 3.6)

We also install matplotlib, pillow 5.4.1 and numpy. Pillow 5.4.1 is needed because of a bug with pillow 6.0.0

    $ wget https://nvidia.box.com/shared/static/j2dn48btaxosqp0zremqqm8pjelriyvs.whl -O torch-1.1.0-cp36-cp36m-linux_aarch64.whl
    $ pip3 install matplotlib pillow==5.4.1 numpy torch-1.1.0-cp36-cp36m-linux_aarch64.whl
    
### Install Torchvision

    $ sudo apt-get install libjpeg-dev zlib1g-dev
    $ git clone -b v0.3.0 https://github.com/pytorch/vision torchvision
    $ cd torchvision
    $ python setup.py install
    $ cd ../  # attempting to load torchvision from build dir will result in import error for _C
    
### Verify the PyTorch and Torchvision installations

    $ python3
    >>> import torch
    >>> print(torch.__version__)
    >>> print('CUDA available: ' + str(torch.cuda.is_available()))
    >>> a = torch.cuda.FloatTensor(2).zero_()
    >>> print('Tensor a = ' + str(a))
    >>> b = torch.randn(2).cuda()
    >>> print('Tensor b = ' + str(b))
    >>> c = a + b
    >>> print('Tensor c = ' + str(c))
    >>> import torchvision
    >>> print(torchvision.__version__)
    
Take note of the version numbers for torch and torch vision. We will need them later. You can exit the interpreter using exit()

## Install Jupyter Lab

Jupyter lab is a browser based IDE-like experience for interactive jupyter notebooks. It will be used to run code on the nano in the browser of the development machine

    $ pip3 install jupyterlab

# Inferencing on the Nano

## Transfer files and Start Jupyter Lab

We need an image file to test the network with. Download an image of an elephant, any image should do. Transfer it to the project directory using scp:

    $ scp ~/Downloads/elephant.jpeg your_jeston_username@jetson_ip:~/your_project_folder

SSH back in the nano, but this time forward port numbers for the jupyter server. This will enable us to access jupyter lab thats running on the nano from the development machine. 

    $ ssh -L 8000:localhost:8888 your_jetson_username@your_jetson_ip
    
We need to start the jupyter server on the nano. Be sure and switch to your virtual env. Since we wont be using a browser on the nano, we can pass in the –no-browser-flag:
 
    $ cd ~/your_project_folder
    $ source env/bin/activate
    $ jupyter lab --no-browser

Copy and paste the link into the address bar of a browser on your development machine. Change the port number from 8888 to 8000 and hit enter. You should see the jupyter lab running with your project files in the left side bar. Create a new python 3 notebook by pressing the python 3 button in the main page.

## Add Classes JSON File to Project Directory

Before we load the model, we are going to need to add the class labels json file. Download it to your development machine by right clicking the following link [classes json file](https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json) and select save link as. You could transfer it using scp, but you can also just add it to the directory using the jupyter lab file browser which is a little easier. Just press the upload files button in the file browser of jupyter lab to upload the downloaded json file. 

## Load the Model 

The following steps are taken  from [here](https://medium.com/@heldenkombinat/image-recognition-with-pytorch-on-the-jetson-nano-fd858a5686aa) and outlined below. 

In a new notebook cell, insert and run the following code. Press shift+enter to run the cell:

```python
import torch, json
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
# Import matplotlib and configure it for pretty inline plots
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```
Load the model in a new cell

```python
model = models.mobilenet_v2(pretrained=True)
# Send the model to the GPU 
model.cuda()
# Set layers such as dropout and batchnorm in evaluation mode
model.eval();
```
We need to load the labels from the json file as well. Run the following in a new cell:

```python
with open("imagenet-simple-labels.json") as f:
    labels = json.load(f)
```

## Load and Transform the Image
We need to add the necessary transformations for the image file so the network gets an input it expects:

```python
data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
```
Go ahead and load the image and plot it

```python
test_image = 'elephant.jpeg'
image = Image.open(test_image)
plt.imshow(image), plt.xticks([]), plt.yticks([])
```

Transform the image using the transformations we created earlier

```
image = data_transform(image).unsqueeze(0).cuda()
```
## Make an Inference

We can see what the network thinks the image is using the code below:

```python
out = model(image)
# Find the predicted class
print("Predicted class is: {}".format(labels[out.argmax()]))
```

## Benchmark

We can run multiple iterations:

```python
import time
fps = np.zeros(200)
with torch.no_grad(): # speed it up by not computing gradients since we don't need them for inference
    for i in range(200):
        t0 = time.time()
        out = model(image)
        fps[i] = 1 / (time.time() - t0)
        
```

and plot the fps results

```python
fig, ax = plt.subplots();  # the semicolon silences the irrelevant output
ax.plot(fps)
ax.set_xlabel("Iteration");
ax.set_ylabel("FPS");
```
        
        
     









