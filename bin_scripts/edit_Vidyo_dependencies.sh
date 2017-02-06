mkdir vidyo.tmp
dpkg-deb -x VidyoDesktopInstaller-ubuntu64-TAG_VD_3_6_3_017 (1).deb ./vidyo.tmp
dpkg-deb --control VidyoDesktopInstaller-ubuntu64-TAG_VD_3_6_3_017 (1).deb ./vidyo.tmp/DEBIAN
# Edit the control file under DEBIAN as per
# http://askubuntu.com/questions/766615/how-to-install-libqt4-core-and-libqt4-gui-on-ubuntu-16-04
dpkg -b ./vidyo.tmp vidyo_modified.deb
sudo dpkg -i vidyo_modified.deb
