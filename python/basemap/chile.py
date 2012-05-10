from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# lat_ts is the latitude of true scale.
# resolution = 'c' means use crude resolution coastlines.
#m = Basemap(projection='poly',lon_0=-70.0,lat_0=-33.0,\
        #llcrnrlon=-35.,llcrnrlat=-30,urcrnrlon=80.,urcrnrlat=50.,lat_ts=20,resolution='c')
m = Basemap(projection='poly',lon_0=-70.0,lat_0=-33.0,\
                    llcrnrlon=-100.,llcrnrlat=-60,urcrnrlon=-30.,urcrnrlat=-10.,lat_ts=20,resolution='c')
#m.etopo()

m.drawcoastlines()
#m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,60.))
#m.drawmapboundary(fill_color='aqua') 
plt.title("Mercator Projection")
plt.savefig('merc.png')
plt.show()
