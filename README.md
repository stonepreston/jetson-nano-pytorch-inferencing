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
