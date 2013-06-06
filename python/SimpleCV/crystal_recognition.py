import SimpleCV
import time
import sys

imagename = sys.argv[1]

img = SimpleCV.Image(imagename)

img = img.smooth()
img = img.dilate()
img = img.erode()

circles = img.findBlobs(minsize=1)

[circle.draw(color=(0,0,255)) for circle in circles]

output = ""
for circle in circles:
  print "-------"
  print circle.coordinates()
  output += "%d %d\n" % (circle.coordinates()[0], circle.coordinates()[1])


newname = "%s_crystal_id.%s" % (imagename.split('.')[-2], imagename.split('.')[-1])
img.save(newname)

newname = "%s_crystal_id_coordinates.txt" % (imagename.split('.')[-2])
outfile = open(newname,'w+')
outfile.write(output)
outfile.close()


img.show()
