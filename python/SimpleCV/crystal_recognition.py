import SimpleCV
import time
import sys

imagename = sys.argv[1]

img = SimpleCV.Image(imagename)

img = img.smooth()
img = img.dilate()
img = img.erode()

circles = img.findBlobs(minsize=1)

[circle.draw(color=(255,255,0)) for circle in circles]

output = ""
for circle in circles:
  print "-------"
  print circle.coordinates()
  output += "%d %d\n" % (circle.coordinates()[0], circle.coordinates()[1])


newname = "marked_images/crystal_id_%s.%s" % (imagename.split('/')[-1].split('.')[-2], imagename.split('/')[-1].split('.')[-1])
img.save(newname)

newname = "crystal_coordinates/crystal_id_coordinates_%s.txt" % (imagename.split('/')[-1].split('.')[-2])
outfile = open(newname,'w+')
outfile.write(output)
outfile.close()


img.show()
