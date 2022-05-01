# Object-Detection-Model

This is a object detection model project.

Let's start assuming we have nothing on my computer, and I will guide you with the installation required for Tensorflow GPU as well.

## 1. Installing anaconda on Windows.
I have used chocolatey to download and install anaconda on my pc. As I want to use Python3 I will open the power shell as administrator and run the command ```choco install anaconda3```. Enter `yes` when prompted. 

## 2. Installing Microsoft Visual C++ Redistributable
Go to [Microsoft website](https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-160) and install the redistributable .exe according to your architecture. My PC is 64 bits and so I have used `X64`. Once downloaded install it.

## 3. Create a Conda Environment
Open Anaconda Prompt as administrator. the terminal will have `(base)` activated by default.
Now in order to get the latest packages that can run tensorflow codes on GPU. Use the [tensorflow website](https://www.tensorflow.org/install/source_windows) and scroll down to Tested Build Configuration `GPU`. Check the lastest python version that can run the GPU.

During my installation process. It was python 3.10, so I created a new conda environment with python 3.10. 
```
conda create -n tf_gpu python==3.10
```
Enter `y` when prompted.

Once the virtual environment is created. Activate the new environment.
```
conda activate tf_gpu
```
