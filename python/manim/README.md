# To run with the manim community docker image
# https://hub.docker.com/r/manimcommunity/manim
docker pull manimcommunity/manim

# To start the docker container
# Give it the name, manimimage
docker run --name manimimage -it --rm -v /home/bellis/matts-work-environment/python/manim:/manim/scenes manimcommunity/manim:latest

# From inside the docker container, for example
# Make sure you're in the /manim directory
manim scenes/hello_circle_37.py SquareToCircle -ql


# To copy files out, for example
docker container ls
docker cp manimimage:/manim/media/videos/hello_circle_37/480p15/SquareToCircle.mp4 .


# Installing to work with Jupyter notebooks

Maybe try with docker?

https://docs.manim.community/en/latest/installation/docker.html

docker run -it -p 8888:8888 manimcommunity/manim jupyter lab --ip=0.0.0.0

## THIS SEEMS TO WORK!
```
conda create -n my-manim-environment jupyter
conda activate my-manim-environment
conda install -c conda-forge pycairo
pip install manim-physics
```



