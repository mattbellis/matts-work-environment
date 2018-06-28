cmake ../root/ \
        -DCMAKE_INSTALL_PREFIX=/usr/local/ \
        -DPYTHON_INCLUDE_DIR=/opt/anaconda3/include/python3.6m \
        -DPYTHON_LIBRARY=/opt/anaconda3/lib/libpython3.6m.so \
        -Dgnuinstall=ON
#sudo cmake --build . --target install -- -j4

