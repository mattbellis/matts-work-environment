from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# lat_ts is the latitude of true scale.
# resolution = 'c' means use crude resolution coastlines.

#country = 'Chile'
country = 'Chile_zoom'

#fig = plt.figure(figsize=(2,10),dpi=200)
fig = plt.figure(figsize=(5,5),dpi=200)

#projection = 'tmerc' # Transverse mercator
projection = 'lcc' # Lambert conformal

resolution='f'

'''
m = Basemap(projection=projection,lon_0=-70.0,lat_0=-33.0,\
            llcrnrlon=-76.,urcrnrlon=-67.,\
            llcrnrlat=-55,urcrnrlat=-10.,\
            lat_ts=20,resolution=resolution)
'''

m = Basemap(projection=projection,lon_0=-70.0,lat_0=-33.0,\
            llcrnrlon=-71.,urcrnrlon=-69.,\
            llcrnrlat=-34,urcrnrlat=-32.,\
            lat_ts=20,resolution=resolution)

style='default'
#m.fillcontinents(color='coral',lake_color='aqua')
#m.drawmapboundary(fill_color='aqua') 

#m.etopo(); style='etopo'
m.warpimage(); style='warpimage'
#m.bluemarble(); style='bluemarble'
#m.shadedrelief(); style="shadedrelief"

m.drawcoastlines()
m.drawcountries()

# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,60.))

plt.title(country)

outname = "%s_res_%s_%s_%s.png" % (country,resolution,projection,style)
plt.savefig(outname)

plt.show()
