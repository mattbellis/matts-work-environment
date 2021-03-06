cd
rm -f bin_scripts
ln -s matts-work-environment/bin_scripts/ .

######### Remove files ############
rm -rf .cshrc
rm -rf .dir_colors
rm -rf .fluxbox/keys
rm -rf .fluxbox/menu
rm -rf ~/matts-work-environment/.hgignore
rm -rf .mrxvtrc
rm -rf .vimrc
rm -rf .hgrc
rm -rf .conkyrc
rm -rf proxy.pac
rm -rf ~/.matplotlib/my_matplotlibrc
rm -rf ~/.Xresources

######### Link files ############
ln -s ~/matts-work-environment/environment_configs/Xresources .Xresources
ln -s ~/matts-work-environment/environment_configs/cshrc .cshrc
ln -s ~/matts-work-environment/environment_configs/dir_colors .dir_colors
ln -s ~/matts-work-environment/environment_configs/fluxbox_keys .fluxbox/keys
ln -s ~/matts-work-environment/environment_configs/fluxbox_menu .fluxbox/menu
ln -s ~/matts-work-environment/environment_configs/hgignore ~/matts-work-environment/.hgignore
ln -s ~/matts-work-environment/environment_configs/mrxvtrc .mrxvtrc
ln -s ~/matts-work-environment/environment_configs/vimrc .vimrc
ln -s ~/matts-work-environment/environment_configs/hgrc .hgrc
ln -s ~/matts-work-environment/environment_configs/conkyrc .conkyrc
ln -s ~/matts-work-environment/environment_configs/proxy.pac proxy.pac
ln -s ~/matts-work-environment/environment_configs/my_matplotlibrc ~/.matplotlib/.


