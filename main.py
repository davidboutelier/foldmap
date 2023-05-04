from functions import make_project_from_file, make_origin, make_lines, make_points_from_lines, make_cmp, make_grid, make_fold_dask, make_fold_gpd
import time


sp = None
rp = None
cmp = None
bins= None

 
# record start time
start = time.time()

# make a project (folder) from the parameter file in current directory
parameter_file = 'parameters.json'
parameters = make_project_from_file(parameter_file=parameter_file)

# make origin shp
make_origin(parameter_file)

# make receiver lines
rl = make_lines(parameter_file=parameter_file, line_type='receiver')

# make source lines
sl = make_lines(parameter_file=parameter_file, line_type='source')

# make receiver points
rp = make_points_from_lines(parameter_file=parameter_file, line_type='receiver', reload_line=False, lines=rl)

# make source  points
sp = make_points_from_lines(parameter_file=parameter_file, line_type='source', reload_line=False, lines=sl)

# make_cmp
cmp = make_cmp(parameter_file=parameter_file, offset=500, to_file=False, reload_points=False, sp=sp, rp=rp, partitions=100)

# make the grid
bins = make_grid(parameter_file=parameter_file, to_file=True)

#bins = make_fold_dask(parameter_file=parameter_file, cmp=cmp, bins=bins, reload=False, to_file=True, partition_cmp=4, partition_bins=4)
bins = make_fold_gpd(parameter_file=parameter_file, cmp=cmp, bins=bins, reload=False, to_file=True, partition=100)

# record end time
end = time.time()
duration = end-start
print(f"operation completed in {duration:.03f}s")