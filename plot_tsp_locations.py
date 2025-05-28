import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

def plot_locations(locations):
    austria_gdf = gpd.read_file("austria_districts.geojson")
    print(locations)
    points = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for _, lat, lon in locations],
        crs="EPSG:4326"
    )

    # Plot the map
    fig, ax = plt.subplots(figsize=(10, 10))
    austria_gdf.plot(ax=ax, color='lightgrey', edgecolor='black')
    points.plot(ax=ax, color='red', markersize=50)

    # Styling
    ax.set_title("Locations of Interest", fontsize=15)
    ax.axis('off')
    plt.show()
