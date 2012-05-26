#./configure linuxx8664gcc \
./configure linux \
                      --prefix=/usr/local \
                      --with-cern-libdir=/usr/lib \
                      --enable-mathcore \
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

