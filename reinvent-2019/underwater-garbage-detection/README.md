# Underwater Garbage Detection - Let's Map It Before We Mop It
Welcome to our re:Invent 2019 builders fair project. This README will guide you through the process of setting the project up by yourself.

Please note that this README just covers the inference and not the training. The training process will be available at a later stage.

## License
This library is licensed under the Apache 2.0 License.

## Prerequisites
_You can run this project on your Laptop or on Nvidia Jetson Xavier / TX2. Jetson Nano is not powerful enough so you can't use it._

This project is using several libraries. Please make sure you install them before proceeding:

| Library | Version | Installation |
|---|---|---|
| Tensorflow | >=1.14 | Please refer to [https://www.tensorflow.org/install/](https://www.tensorflow.org/install/) |
| Python | >= 3.6.x | Please install according to your operating system |
| Pip | same version as python | Please install according to your operating system |
| OpenCV | | 

## Installation
### Running locally on you laptop
Please make sure you install all necessary libraries mentioned in Prerequisites before you proceed.
#### Install necessary python libraries.
We recommend to use a python virtual environment so that you have a clean environment for this project.
First we want to have the missing libraries installed, so please open a command line and execute the following command (make sure to use python3 / pip3):

```$ pip install -r requirements.txt```

This should install all necessary libraries so that you can start with the project