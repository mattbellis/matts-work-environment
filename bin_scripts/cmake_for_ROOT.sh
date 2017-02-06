cmake ../root/ \
        -DCMAKE_INSTALL_PREFIX=/usr/local/ \
        -DPYTHON_INCLUDE_DIR=/opt/anaconda3/include/python3.5m \
        -DPYTHON_LIBRARY=/opt/anaconda3/lib/libpython3.5m.so \
        -Dgnuinstall=ON
#sudo cmake --build . --target install -- -j4

