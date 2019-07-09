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

    $ wget https://nvidia.box.com/shared/static/j2dn48btaxosqp0zremqqm8pjelriyvs.whl -O torch-1.1.0-cp36-cp36m-linux_aarch64.whl
    $ pip3 install numpy torch-1.1.0-cp36-cp36m-linux_aarch64.whl
    
### Install Torchvision

    $ sudo apt-get install libjpeg-dev zlib1g-dev
    $ git clone -b v0.3.0 https://github.com/pytorch/vision torchvision
    $ cd torchvision
    $ sudo python setup.py install
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

