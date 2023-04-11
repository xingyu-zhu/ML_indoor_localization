from pyproj import Transformer

transformer = Transformer.from_crs("epsg:32730", "epsg:4326")

x = 4864806.3498992985
y = -7341.616870573757

lat, long = transformer.transform(x, y)
print(lat, long)