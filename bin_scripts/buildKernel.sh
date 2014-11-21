#!/bin/tcsh -f

set topdir = $1
set version = $2

cd $topdir
sudo make ARCH=i386 clean 
sudo make ARCH=i386 bzImage 
ls -l arch/i386/boot
sudo make ARCH=i386 modules
sudo make ARCH=i386 modules_install

sudo mkinitrd -o /boot/initrd-$version.img $version
sudo cp arch/i386/boot/bzImage /boot/bzImage-$version
sudo cp System.map /boot/System.map-$version
sudo rm /boot/System.map
sudo ln -s /boot/System.map-$version /boot/System.map
                      


