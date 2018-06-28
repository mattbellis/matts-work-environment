<<<<<<< HEAD
./configure \
                      --prefix=/usr/local \
                      --enable-mathmore \
                      --enable-python \
                      --enable-minuit2 \
                      --disable-xrootd \
                      --enable-roofit \
                      --enable-fftw3 \
                      --disable-builtin-llvm \
                      --enable-soversion \
                      --enable-table 
=======
cmake ../root/ -DCMAKE_INSTALL_PREFIX=/usr/local/ -DPYTHON_INCLUDE_DIR=/home/bellis/anaconda2/include/python2.7 -DPYTHON_LIBRARY=/home/bellis/anaconda2/lib/libpython2.7.so -Dgnuinstall=ON

#./configure \
                      #--prefix=/usr/local \
                      #--enable-mathmore \
                      #--enable-python \
                      #--enable-minuit2 \
                      #--disable-xrootd \
                      #--enable-roofit \
                      #--enable-fftw3 \
                      #--disable-builtin-llvm \
                      #--enable-soversion \
                      #--enable-table 
>>>>>>> bb0e4bd0e440e91b660b8bd0d5ccee8fad263980




#echo Dont forget to unsetenv TOP_DIR

