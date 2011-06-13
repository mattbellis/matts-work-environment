#!/bin/tcsh 

sudo apt-get install nvidia-kernel-common module-assistant
sudo m-a -i -t prepare
#sudo apt-get remove nvidia-glx nvidia-glx-dev nvidia-kernel-source
sudo apt-get remove nvidia-glx nvidia-kernel-source
sudo apt-get install  nvidia-kernel-source/unstable
sudo m-a clean,a-i -i -t -f nvidia-kernel-source
sudo depmod -a
#sudo apt-get install nvidia-glx/unstable nvidia-glx-dev/unstable nvidia-xconfig/unstable xserver-xorg/unstable
sudo apt-get install nvidia-glx/unstable nvidia-xconfig/unstable xserver-xorg/unstable
sudo dpkg-reconfigure xserver-xorg

