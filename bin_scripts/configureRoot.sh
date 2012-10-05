#./configure linuxx8664gcc \
./configure linux \
                      --prefix=/usr/local \
                      --with-x11-libdir=/usr/lib/x86_64-linux-gnu/ \
                      --enable-mathmore \
                      --enable-python \
                      --enable-minuit2 \
                      --fontdir=/usr/share \
                      --disable-xrootd \
                      --enable-roofit \
                      --enable-fftw3 \
                      --enable-soversion \
                      --enable-table 



#echo Dont forget to unsetenv TOP_DIR

