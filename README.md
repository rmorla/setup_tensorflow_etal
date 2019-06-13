# setup_tensorflow_etal


## A. Base system -- ubuntu 18.04 LTS server -- http://releases.ubuntu.com/18.04/

## B. Install CUDA software to access the GPU

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

lspci | grep -i nvidia
uname -m && cat /etc/*release
gcc --version
sudo apt-get install gcc
sudo apt-get install linux-headers-$(uname -r)



## C. Docker

https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04

## D. Tensorflow on Docker 

https://www.tensorflow.org/install/docker
