### yodiipy GitHub Pages site

The *yodiipy* library is a collection of multi-purpose Python libraries. Some notable components include:
- contours2kml.py: Convert matplotlib.pyplot contours to KML. This library uses a knot-theory inspired approach to follow contours and then identify "pinch points," where -- from pattern of the sequence, we infer that we have closed a contour and are switching to a new level. There are other approaches, to use built in meta-data, but those appear to be more susceptible to updates in matplotlib. This method has survived several upgrades to the plotting routine, and we keep our fingers crossed.
- ANSStools.py: A collection of tools to access earthquake catalogs. Recent updates include a simple web-query API to the USGS/ANSS ComCat catalog.

