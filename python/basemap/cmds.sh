avconv -delay 10 -f image2 -i merc_with_data_%04d.png foo.avi

mogrify -resize 800x800 *.jpg
ffmpeg -r 25 -qscale 2 -i merc_with_data_%04d.jpg output.mp4

http://www.itforeveryone.co.uk/image-to-video.html
ffmpeg -r 24 -qscale 1 -i %05d.morph.jpg output.mp4
