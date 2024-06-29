import pandas as pd
import plotly.express as px

metadata = pd.read_csv("../metadata/metadata.csv")

print("\nNumber of images from each disaster:")
print(metadata["disaster"].value_counts())

print("\nTime difference between the capture of the pre- and post-flooding satellite images:")
print(metadata.groupby("disaster")["days_difference"].value_counts())

print("\nDate of the capture of the post-flooding satellite images:")
print(metadata.groupby("disaster")["post_date"].value_counts())

print("\nLocation of the satellite images: (in browser)")
fig = px.scatter_mapbox(metadata, lat="y_min",lon="x_min", hover_name="image", zoom=1, height=800, width=1200)
fig.update_layout(mapbox_style="open-street-map");
fig.show()