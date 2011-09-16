#!/bin/tcsh 

###############################
# Working prior to July, 2011
# 
# Now failing on install of nvidia-glx
#
###############################
#sudo apt-get install nvidia-kernel-common module-assistant
#sudo m-a -i -t prepare
#sudo apt-get remove nvidia-glx nvidia-kernel-source
#sudo apt-get install  nvidia-kernel-source/unstable
#sudo m-a clean,a-i -i -t -f nvidia-kernel-source
#sudo depmod -a
#sudo apt-get install libgl1-nvidia-glx/unstable
#sudo apt-get install nvidia-glx/unstable nvidia-xconfig/unstable xserver-xorg/unstable
#sudo dpkg-reconfigure xserver-xorg

# For laptop
#sudo apt-get remove nvidia-glx nvidia-glx-dev nvidia-kernel-dkms
#sudo apt-get install nvidia-kernel-dkms
#sudo apt-get install nvidia-vdpau-driver/unstable
#sudo apt-get install nvidia-glx/unstable libgl1-nvidia-glx/unstable nvidia-xconfig/unstable xserver-xorg/unstable
#sudo nvidia-xconfig

# For desktop
# Seems to work with 2.6.32
# nvidia driver 280
sudo apt-get remove nvidia-glx nvidia-glx-dev nvidia-kernel-dkms
sudo apt-get install nvidia-kernel-dkms/unstable
sudo apt-get install nvidia-vdpau-driver/unstable
#sudo apt-get install nvidia-glx/unstable nvidia-glx-dev/unstable libgl1-nvidia-glx/unstable nvidia-xconfig/unstable xserver-xorg/unstable
sudo apt-get install nvidia-glx/unstable libgl1-nvidia-glx/unstable nvidia-alternative/unstable nvidia-xconfig/unstable xserver-xorg/unstable
#nvidia-kernel-280.13/unstable 
sudo nvidia-xconfig

