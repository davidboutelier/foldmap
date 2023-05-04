# foldmap

## Aim
Calculate the surface fold for 3D seismic survey and produce shp files to be plotted in QGIS.

## How does it work 
Functions are in module function.py
Call the the functions in the main python module main.py or call them in GUI (TO DO).

Yaml file is provided to get conda env.

parameters of the seismic survey can be entered in json file

# WORKING
- create project
- create lines
- create points from lines
- create cmp with multiple small blocks to avoid memory 
- make bins
- calculate fold in bins with geopandas

# TO DO
- make cmp with dask geopandas
- finish making fold with dask geopandas for large surveys
- stand-alone GUI
- QGIS plug-in
- save parameters in project folder
- add vertical stacking?
- add distribution of azimuths?
- minimum offset in bins?
