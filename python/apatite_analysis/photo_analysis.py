import SimpleCV
import time
from subprocess import call

def saveFilmToDisk(bufferName, outname):
    # construct the encoding arguments
    params = " -i {0} -c:v mpeg4 -b:v 700k -r 24 {1}".format(bufferName, outname)
    # run avconv to compress the video since ffmpeg is deprecated (going to be).
    call('avconv'+params, shell=True)


js = SimpleCV.JpegStreamer()

BUFFER_NAME = 'buffer.avi'

# create the video stream for saving the video file
vs = SimpleCV.VideoStream(fps=24, filename=BUFFER_NAME, framefill=True)


img = SimpleCV.Image('gvg_ft_photo2.jpg')
img = img.smooth()
img = img.dilate()
img = img.erode()
img = img.invert()
#lines = img.findLines(threshold=25,minlinelength=20,maxlinegap=20)
#lines = img.findLines(threshold=90,minlinelength=50,maxlinegap=10)
lines = img.findBlobs(maxsize=50000,appx_level=1,minsize=10)
[line.draw(color=(0,255,0)) for line in lines]
sum = 0
icolor = 255.0/len(lines)
'''
for i,line in enumerate(lines):
    #print "%f %f" % (line.x,line.y)
    #line.draw(color=(255-i*icolor,i*icolor,0))
    img.draw(line)
    line.image = img
    #print "-------"
    #print line.coordinates()
    #print line.points
    #print line.length()
    #sum = line.length() + sum
    #print sum / len(lines)
    img.show()
    vs.writeFrame(img)
'''

print "N blobs: %d" % (len(lines))
#img.save(js.framebuffer)
img.save("processed_image.png")

#saveFilmToDisk(BUFFER_NAME, "blob_finding_movie.mp4")



'''
while 1:
    img.show()
'''
