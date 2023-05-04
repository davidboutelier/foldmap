import json
from tqdm import tqdm
from shapely.geometry import Polygon, Point, MultiPoint, LineString, MultiLineString
import numpy as np
import geopandas as gpd
import os
import shutil
import time
from scipy.spatial.distance import cdist
import dask_geopandas
import pandas as pd

def make_archive(source, destination):
    base_name = '.'.join(destination.split('.')[:-1])
    format = destination.split('.')[-1]
    root_dir = os.path.dirname(source)
    base_dir = os.path.basename(source.strip(os.sep))
    shutil.make_archive(base_name, format, root_dir, base_dir)

def make_project_from_file(parameter_file):
    '''
    Generates the dictionary for the project from a parameter file
    '''
    print('creating project from parameters...')
    with open(parameter_file) as f:
        parameters = json.load(f)

    return parameters

def make_origin(parameter_file):

    print('creating origin point...')
    with open(parameter_file) as f:
        parameters = json.load(f)
    
    rootfolder = parameters['rootfolder']
    job_number = parameters['job_number']
    client = parameters['client']
    project = parameters['project']
    option = parameters['option']
    epsg = parameters['epsg']
    origin = parameters['origin']

    point  = []
    point.append(Point([origin[0], origin[1]]))

    origin_point = gpd.GeoDataFrame(geometry=point, crs='epsg:'+str(epsg))
    if os.path.exists(os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, 'origin')):
        pass
    else:
        os.makedirs(os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, 'origin'))

    origin_point.to_file(os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, 'origin', 'origin.shp'))
    make_archive(os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, 'origin'), os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, 'origin.zip'))
    shutil.rmtree(os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, 'origin'))

def make_lines(parameter_file, line_type, add_shift):

    print('creating '+line_type+' lines...')
    with open(parameter_file) as f:
        parameters = json.load(f)

    rootfolder = parameters['rootfolder']
    job_number = parameters['job_number']
    client = parameters['client']
    project = parameters['project']
    option = parameters['option']
    epsg = parameters['epsg']
    origin = parameters['origin']
    
    if line_type == "receiver":
        
        line_length = parameters['receiver_line_length']
        line_spacing = parameters['receiver_line_spacing']
        line_azimuth = parameters['receiver_line_azimuth']
    
        n_lines = parameters['receiver_line_number']
        line_file = parameters['receiver_lines_file'] 
        shifts = parameters['receiver_line_shift']
        crossline_shift = shifts[1]
        inline_shift = shifts[0]

    elif line_type=='source':
        line_length = parameters['source_line_length']
        line_spacing = parameters['source_line_spacing']
        line_azimuth = parameters['source_line_azimuth']
    
        n_lines = parameters['source_line_number']
        line_file = parameters['source_lines_file'] 
        shifts = parameters['source_line_shift']
        crossline_shift = shifts[1]
        inline_shift = shifts[0]

    lines = []

    n=n_lines
    for i in tqdm(range(0,n)):
        xs = origin[0] + add_shift[0] + (crossline_shift  + i*line_spacing ) * np.sin((360 + 90-line_azimuth)%360 * np.pi/180) + inline_shift * np.sin((360+line_azimuth)%360 * np.pi/180)
        ys = origin[1] + add_shift[1] - (crossline_shift + i* line_spacing ) * np.cos((360 + 90-line_azimuth)%360 * np.pi/180) + inline_shift * np.cos((360+line_azimuth)%360 * np.pi/180)

        xe = xs + (line_length) * np.sin((360+line_azimuth)%360 * np.pi/180)
        ye = ys + (line_length) * np.cos((360+line_azimuth)%360 * np.pi/180)
        line = LineString([(xs,ys), (xe, ye)])
        lines.append(line)

    lines_gpd = gpd.GeoDataFrame(geometry=lines, crs='epsg:'+str(epsg))

    k = 1
    while os.path.exists(os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, line_type+str(k)+'_lines.zip')):
        k = k+1 

    os.makedirs(os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, line_type+str(k)+'_lines'))

    lines_gpd.to_file(os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, line_type+str(k)+'_lines', line_type+'_lines.shp'))
    make_archive(os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, line_type+str(k)+'_lines'), os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, line_type+str(k)+'_lines.zip'))
    shutil.rmtree(os.path.join(rootfolder, str(job_number)+'-'+client+'-'+project, option, line_type+str(k)+'_lines'))


    return lines_gpd

def make_points_from_lines(parameter_file, line_type, reload_line, lines):
    print('creating '+line_type+' points...')

    with open(parameter_file) as f:
            parameters = json.load(f)

    if line_type=='receiver':
        point_spacing = parameters["receiver_point_spacing"]
        line_file = "receiver_lines"

    else:
        point_spacing = parameters["source_point_spacing"]
        line_file = "source_lines"
    
    if reload_line:

        # unzip the lines file
        shutil.unpack_archive(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'], line_file+'.zip'), 
                                        os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp'))
        #read the lines file as gpd
        gdf = gpd.read_file(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp', line_file, line_file+'.shp'))
        
        # delete the temporary file
        shutil.rmtree(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp'))
    else:
        gdf = lines
    
    shift = 0 # shift the lines not the points
    line_number = 0
    x_points = []
    y_points = []
    these_lines = gdf.geometry
    for line in tqdm(these_lines):
        line_number=line_number+1
        xy = np.array(line.coords.xy).T
        n_points = xy.shape[0]
        left = 0

        for i in range(0, n_points-1):
            if i==0:
                x_points.append(xy[i,0])
                y_points.append(xy[i,1])
            seg_len = np.sqrt(np.power((xy[i,0] -xy[i+1,0]),2) + np.power((xy[i,1] - xy[i+1, 1]),2))

            if (seg_len + left) > point_spacing:
                seg_n_points = int(np.floor((seg_len - shift + left) / point_spacing))
                newleft = seg_len + left -seg_n_points * point_spacing
                for j in range(0, seg_n_points):
                    theta = np.arctan2(xy[i+1, 1] -xy[i, 1], xy[i+1, 0] -xy[i,0])
                    dx = (point_spacing + shift -left) * np.cos(theta) + j * point_spacing * np.cos(theta)
                    dy = (point_spacing + shift -left) * np.sin(theta) + j * point_spacing * np.sin(theta)
                    x_points.append(xy[i,0] + dx)
                    y_points.append(xy[i,1] + dy)
                left = np.copy(newleft)
            else:
                left = left + seg_len
    points = [Point(sx,sy) for sx, sy in zip(x_points, y_points)]  
    points_df = gpd.GeoDataFrame(geometry=points, crs='epsg:'+str(parameters['epsg']))

    if parameters['clip_outline']:
        print('clipping points...')
        
        # unzip the lines file
        shutil.unpack_archive(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'], parameters['clip_file_outline']+'.zip'), 
                                        os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp'))
        
        #read the lines file as gpd
        outline_gdf = gpd.read_file(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp', parameters['clip_file_outline'], parameters['clip_file_outline']+'.shp'))
        
        # delete the temporary file
        shutil.rmtree(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp'))
        
        clipped_points = points_df.clip(outline_gdf)
        points = clipped_points
    else:
        points = points_df

    print('saving file...')
    k=1
    while os.path.exists(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], line_type+str(k)+'_points.zip')):
        k = k+1
    #if os.path.exists(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], line_type+'_points')):
    #    pass
    #else:
    
    os.makedirs(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], line_type+str(k)+'_points'))

    points.to_file(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], line_type+str(k)+'_points', line_type+str(k)+'_points.shp'))
    make_archive(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], line_type+str(k)+'_points'), 
                 os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], line_type+str(k)+'_points.zip'))
    shutil.rmtree(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], line_type+str(k)+'_points'))
    
    print(str(len(points)) + ' '+ line_type + ' points created')

    return points
    
    
    
def make_cmp(parameter_file, offset, to_file, reload_points, sp, rp, partitions):
    print('creating the midpoints...')
    n_chunks=partitions
    with open(parameter_file) as f:
        parameters = json.load(f)

    if reload_points:
        print('reloading the source and receiver points...')
        #load the receiver points
        shutil.unpack_archive(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'], 'receiver_points.zip'), 
                                        os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp'))
        #read the lines file as gpd
        receivers = gpd.read_file(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                            parameters['option'],'temp', 'receiver_points', 'receiver_points.shp'))
        
        # delete the temporary file
        shutil.rmtree(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                            parameters['option'],'temp'))

        #load the receiver points
        shutil.unpack_archive(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'], 'source_points.zip'), 
                                        os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp'))
        #read the lines file as gpd
        sources = gpd.read_file(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                            parameters['option'],'temp', 'source_points', 'source_points.shp'))
        
        # delete the temporary file
        shutil.rmtree(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                            parameters['option'],'temp'))
    else:
        receivers=rp
        sources=sp

    print('calculating the positions of the midpoints...')
    # evaluate the pairs
    r = []
    for p in receivers.geometry:
        r.append([p.x, p.y])
    nr = np.array(r)

    s = []
    for p in sources.geometry:
        s.append([p.x, p.y])
    ns = np.array(s).copy()

    if partitions == 1:
        dist = cdist(ns, nr, 'euclidean')
        indices_S, indices_R = np.where(dist < offset)
        s2 = np.copy(ns[indices_S, :], order='F').astype('float32')
        r2 = np.copy(nr[indices_R, :], order='F').astype('float32') 
        points = s2 + 0.5 * (r2 - s2)
        all_cmp = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=points[:,0], y=points[:,1]), crs = 'EPSG:'+str(parameters['epsg']))

    else:

        os.makedirs(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp'))
    
        chunk = int(len(s) / n_chunks)

        for i in tqdm(range(0,n_chunks)):
            
            if i == 0:
                sub = ns[:chunk].copy()
            elif i == n_chunks-1:
                sub = ns[i * chunk :].copy()
            else:
                sub = ns[i * chunk :(i + 1) * chunk].copy()

            dist = cdist(sub, nr, 'euclidean')
            indices_S, indices_R = np.where(dist < offset)

            s2 = np.copy(sub[indices_S, :], order='F').astype('float32')
            r2 = np.copy(nr[indices_R, :], order='F').astype('float32')
            
            points = s2 + 0.5 * (r2 - s2)

            
            np.savetxt(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp', 'temp_'+str(i)+'.csv'), points, delimiter=",")
            del s2, r2, points   
        
        print('assembling the points...')
        for i in tqdm(range(0,n_chunks)):
            filename = os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'],'temp', 'temp_'+str(i)+'.csv')
            arr = np.genfromtxt (filename, delimiter=",")
            this_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=arr[:,0], y=arr[:,1]), crs = 'EPSG:'+str(parameters['epsg']))
            del arr
            if i ==0:
                all_cmp = this_gdf.copy()
            else:
                all_cmp = pd.concat([all_cmp, this_gdf])
        
        shutil.rmtree(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'temp'))

    print(str(int(len(all_cmp)))+' midpoints in survey area for offset '+str(offset)) 

    if to_file:
        print('saving the files...')
        if os.path.exists(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'cmp')):
            pass
        else:
            os.makedirs(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'cmp'))

        all_cmp.to_file(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'cmp', 'cmp.shp'))
        make_archive(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'cmp'), 
                    os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'cmp.zip'))
        shutil.rmtree(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'cmp'))

    return  all_cmp   

def make_grid(parameter_file, to_file):
    print('creating the grid...')
    with open('parameters.json') as f:
        parameters = json.load(f)

    origin = parameters['origin']
    polygons = []

    nbin_inline = int(np.ceil(parameters['survey_area_inline_length'] / parameters['bin_length_inline']))+1
    nbin_crossline = int(np.ceil(parameters['survey_area_crossline_length'] / parameters['bin_length_crossline']))+1

    dxxb = parameters['bin_length_inline'] * np.cos((360 + 90-parameters['survey_area_inline_azimuth'])%360 *np.pi/180) 
    dxyb = parameters['bin_length_inline'] * np.sin((360 + 90-parameters['survey_area_inline_azimuth'])%360 *np.pi/180) 

    dyxb = parameters['bin_length_crossline'] * np.cos((360 + parameters['survey_area_inline_azimuth'])%360 *np.pi/180)
    dyyb = parameters['bin_length_crossline'] * np.sin((360 + parameters['survey_area_inline_azimuth'])%360 *np.pi/180)

    nbin = nbin_inline*nbin_crossline
    print(str(int(nbin)) + ' bins in the survey area')

    for i in tqdm(range(0,nbin_inline)):
        for j in range(0, nbin_crossline):
            a = np.array([origin[0] + i * dxxb + j* dyxb, origin[1] + j * dyyb + i * dxyb])
            b = a + np.array([dxxb, dxyb])
            c = b + np.array([dyxb, dyyb])
            d = a + np.array([dyxb, dyyb])

            polygons.append(Polygon([a,b,c,d]))

    bins = gpd.GeoDataFrame(geometry=polygons, crs='epsg:'+str(parameters['epsg']))
    bins['fold'] = 0

    if to_file:
        print('saving file...')
        if os.path.exists(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins')):
            pass
        else:
            os.makedirs(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins'))

        bins.to_file(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins', 'bins.shp'))

        make_archive(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins'), 
                    os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins.zip'))
        shutil.rmtree(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins'))


    return bins

def make_fold_dask(parameter_file, cmp, bins, reload, to_file, partition_cmp, partition_bins):

    with open(parameter_file) as f:
        parameters = json.load(f)


    print(bins)

    print('converting cmp geodataframe to dask geodataframe...')
    dpoints =  dask_geopandas.from_geopandas(cmp, npartitions=partition_cmp)

    print('converting bins geodataframe to dask geodataframe...')
    dbins = dask_geopandas.from_geopandas(bins, npartitions=partition_bins)

    print('reset and rename index bins...')
    dbins_reindexed = dbins.reset_index().rename(columns={'index': 'bins_index'}).compute()
    #dbins_reindexed = dbins.rename(columns={'index': 'bins_index'}).compute()
    print(dbins_reindexed)

    print('spatially join bins and cmps...')
    joined_dgpd = dask_geopandas.GeoDataFrame.sjoin(dbins_reindexed, dpoints, how='inner', predicate='intersects').compute()
    print(joined_dgpd)

    print('group by bin index...')
    bin_stats = joined_dgpd.groupby('bins_index')
    print(bin_stats.head())


    bin_stats2 = bin_stats.agg('size')
    
    print(bin_stats2)
    
    print('assembling...')
    for i in tqdm(range(0,len(bin_stats2))):
        bins.at[i, 'fold'] = bin_stats2.iloc[i]
        #print(bins.at[i, 'fold'])
        #print(bin_stats.iloc[i])
        #bins.at[i, 'fold'] = bin_stats[i]
    
    if to_file:
        print('saving file...')
        if os.path.exists(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins')):
            pass
        else:
            os.makedirs(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins'))

        bins.to_file(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins', 'bins.shp'))

        make_archive(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins'), 
                    os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins.zip'))
        shutil.rmtree(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins'))
    


    return bins

def make_fold_gpd(parameter_file, cmp, bins, reload, to_file, partition, do_azimuth):

    print('calculating the fold...')

    with open(parameter_file) as f:
        parameters = json.load(f)

    n_cmp = len(cmp)
    chunk = int(n_cmp / partition)

    b = bins.copy()

    for i in tqdm(range(0, partition)):
        if i == 0:
            m0 = cmp[:chunk].copy()
        elif i == partition - 1:
            m0 = cmp[(i * chunk) :].copy()
        else:
            m0 = cmp[(i * chunk) :(i + 1) * chunk].copy()

        reindexed = b.reset_index().rename(columns={'index': 'bins_index'})
        joined = gpd.tools.sjoin(reindexed, m0, predicate='contains')
                
        bin_stats = joined.groupby('bins_index').agg({'fold': len})

        arr = np.array(bin_stats.index)

        for k in range(len(arr)):
            index = arr[k]
            fold = int(bin_stats.iloc[k, 0])
            bins.loc[index, 'fold'] = bins.loc[index, 'fold'] + fold
            bins.loc[index, 'fold']

    if parameters['clip_outline']:

        # unzip the lines file
        shutil.unpack_archive(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'], parameters['clip_file_outline']+'.zip'), 
                                        os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp'))
        
        #read the lines file as gpd
        outline_gdf = gpd.read_file(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp', parameters['clip_file_outline'], parameters['clip_file_outline']+'.shp'))
        
        # delete the temporary file
        shutil.rmtree(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], 
                                        parameters['option'],'temp'))
        clipped_bins = bins.clip(outline_gdf)
        bins = clipped_bins

    if to_file:
        print('saving file...')
        if os.path.exists(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins')):
            pass
        else:
            os.makedirs(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins'))

        bins.to_file(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins', 'bins.shp'))

        make_archive(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins'), 
                    os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins.zip'))
        shutil.rmtree(os.path.join(parameters['rootfolder'], str(parameters['job_number'])+'-'+parameters['client']+'-'+parameters['project'], parameters['option'], 'bins'))

    return bins