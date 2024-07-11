# Thursday, Jul 11: defining a climate index

## Overview:
Part 1. [Set filepaths and import packages](#Part-1:-Setting-filepaths-and-importing-packages)  
Part 2: [Open the ERA5 temperature data](#Part-2:-Open-the-data)  
Part 3: [Compute a climate index (monthly temperature anomaly in Woods Hole)](#Part-3:-Defining-the-index)

## Goal figures:
<p float="left">
 <img src="../readme_figs/task_list_t2m_sample.png" width="400" />
</p>

<p float="left">
 <img src="../readme_figs/task_list_WH_anomaly.png" width="300" />
</p>

## Part 1: Setting filepaths and importing packages
1. __Open the [0_xarray_tutorial.ipynb](scripts/0_xarray_tutorial.ipynb) notebook__. You can do this using your own virtual environment or in the cloud using Google Colab (see [instructions for Colab](#Google-Colab-instructions)).
2. __Run the 1<sup>st</sup> code cell__ (under the header "Check if we're running in Google Colab"):
```
try:
    import google colab
    ...
    IN_COLAB = True

except:
    IN_COLAB = False

```
The purpose of this cell is to detect if we're using Google Colab and, if so, link the notebook with Google Drive. If you're using Colab, you may get an error message about the kernel crashing after the cell finishes executing – if so, run the cell again; it seems to work for me the second time.

3. __Update the filepath ```era_t2m_path``` in the 2<sup>nd</sup> code cell.__ This is the path to the 2m-temperature data from the ERA5 reanalysis. If you're using Colab (```IN_COLAB``` is True), you shouldn't have to change anything. If you're not using Colab, you need to modify the filepath to match the location of the data (if you're on Windows, you may have issues with backslashes; see [note below](#Windows-filepaths)).
```
if IN_COLAB:

    ### This is the path to use with Google Colab.
    ### You shouldn't need to change it.
    era5_t2m_path = ...

else:

    ### Modify this path if you're *not* on Google Colab.
    ### It should be the path to the data on your computer.
    era5_t2m_path = ...
```

3b. (optional) Update the other filepaths in the 2<sup>nd</sup> code cell, ```era5_slp_path``` and ```miroc6_path```. This will allow you to run all of the code in the notebook (e.g., illustrating correlation between variables and regridding), but isn't necessary for today's goal.

4. __Run the 2<sup>nd</sup> code cell__ (filepaths, from above)  and __3<sup>rd</sup> code cell__ (package imports)

5. (optional) Run the rest of the code in the notebook.

### Part 2: Open the data
6. Scroll down to the end of the notebook, and __create a new code cell__.
7. __Get the "pattern" of filenames in the ERA5 dataset__ (there's one file per year). One way to do this is ```file_pattern = os.path.join(era5_t2m_path, "*.nc")```. This pattern will match any files in the ERA5 folder which end in ```.nc``` (file extension for NetCDF files).  To check this worked, you can print out a list of matched files with ```glob.glob(file_pattern)```.
8. __Open the dataset__, with ```t2m = xr.open_mfdataset(file_pattern)["t2m"]```.
9. __Plot the first timestep__, with ```plt.contourf(t2m.longitude, t2m.latitude, t2m.isel(time=0))```


### Part 3: Defining the index
9. __Get 2m temperature near Woods Hole__ with ```t2m_WH = t2m.sel(latitude=41.5, longitude=288.5)```
10. __Define climate index as temperature anomaly in Woods Hole__: first, compute the monthly averages. Then subtract the monthly averages from the original timeseries. For example,
```python
t2m_WH_monthly_avg = t2m_WH.groupby("time.month").mean()
climate_index = t2m_WH.groupby("time.month") - t2m_WH_monthly_avg
```
  
11. (optional): write a function which combines steps 9 and 10:
```python
    def get_climate_index(global_t2m):
        """
        Input: 'global_t2m' is a 3-D DataArray (lon,lat,time)
    
        Output: 'climate_index' is a 1-D DataArray (time)
        """
    
        ## To-do
        climate_index = ...
    
        return climate_index
```
12. __Plot the result__ with ```plt.plot(climate_index.time, climate_index)```





## Windows filepaths
Python reads backslashes (```\```) as escape characters. To avoid issues when specifying filepaths, you could try the ```Path``` function from the ```pathlib``` module (see this [blog post](https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f) for some examples). The idea is to use only *forward* slashes (```/```) in the string; the ```Path``` function will convert the string to the correct form. For example, if the Windows filepath to the data uses *back* slashes, "C:\data\file.nc", you would use *forward* slashes in the string you pass to ```pathlib.Path```:
```
import pathlib
era5_t2m_path = pathlib.Path("C:/data/file.nc")
```
 
## Google Colab instructions
- Go to [https://colab.research.google.com/](https://colab.research.google.com/)
- In Google Colab, click "File -> Open notebook -> Upload" and select one of the tutorial notebooks. 
- Go to the [shared Google Drive folder](https://drive.google.com/drive/folders/1V-aHzoYYUrU6d5ExnxEORiZxeAAisx_e?usp=drive_link)
- Click the three dots next to the "climate-data" folder and click "Organize -> Add shortcut", then select "My Drive"
- Note: __if using Google Colab, the regridding components of the tutorial will not work__, owing to package compatibility issues (cannot import the ```xesmf``` package in Colab, possibly related to [this issue](https://github.com/conda-forge/esmf-feedstock/issues/91)).