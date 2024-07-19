# Climate data analysis @ WHOI: a tutorial
The purpose of this tutorial is to provide practical guidance on how to analyze gridded climate data stored on WHOI's servers using python. The tutorial is designed to take place over ~6 hour-long sessions, and is designed to cover four topics: (1) data pre-processing, (2) model validation, (3) using pre-industrial control runs to detect a climate change signal, and (4) model intercomparison. In addition to the [tutorials](scripts/tutorials) we've provided several additional [examples](scripts/examples) which illustrate the use of large ensembles and paleo-proxies, for example. Below, you will find [recent announcements](#71924-updates), a [description of the individual tutorial sessions](#Outline-for-summer-2024-tutorial) and [other examples](#Description-of-other-examples), [setup instructions](#Setup), an [overview of the project structure](#Description-of-high-level-folders--files), and [other potentially useful links](#Other-potentially-useful-links). Before going into these details, here's a preview:

### Pre-processing with ```xarray```
from [xarray_reference.ipynb](scripts/xarray_reference.ipynb)
<p float="left">
 <img src="./readme_figs/wh_corr.png" width="400" />
</p>

### Model validation
__2m-temperature bias CESM2 (relative to ERA5)__, from [model_validation.ipynb](scripts/tutorials/model_validation.ipynb)
<p float="left">
 <img src="./readme_figs/cesm2_bias.png" width="500" />
</p>

__Gulf Stream bias in CESM2 (relative to ORAS5)__, from [T2m_and_gulfstream_validation.ipynb](scripts/examples/T2m_and_gulfstream_validation.ipynb)
<p float="left">
 <img src="./readme_figs/gulf_stream_comparison.png" width="500" />
</p>

### Climate change detection
__Woods Hole 2m-temperature compared to PI control (CESM2)__, from [climate_change_detection.ipynb](scripts/tutorials/climate_change_detection.ipynb)
<p float="left">
 <img src="./readme_figs/task_list_cesm_histogram.png" width="350" />
</p>

### Intermodel comparison and large ensembles
__Woods Hole 2m-temperature projections (1\% year<sup>-1</sup> CO<sub>2</sub> scenario)__, from ([intermodel_comparison.ipynb](scripts/tutorials/intermodel_comparison.ipynb))
<p float="left">
 <img src="./readme_figs/task_list_1pctCO2.png" width="700" />
</p>


__1,000-member toy model ensemble__ (fig. from ([stochastic_large_ensemble.ipynb](scripts/examples/stochastic_large_ensemble.ipynb))
<p float="left">
 <img src="./readme_figs/ensemble_mean.png" width="350" />
</p>

## 7/19/24 updates
- The project structure (including filenames) has changed since the last tutorial! The [outline below](#Outline-for-summer-2024-tutorial) contains links to notebooks that we went through together in class. 
- Rendered results from the tutorial notebooks can be found in the [results](results) folder. To see the results for a given notebook, click through to the folder with the notebook's name, then open the markdown (".md") file inside.
- See [below](#Other-potentially-useful-links) for links to (i) the CMIP6 overview paper and (ii) CMIP naming conventions

## Outline for summer 2024 tutorial
Date | Topic | Notebook
-- | -- | --
7/9/24 | Connecting to the [CMIP5](cmip5.whoi.edu)<sup>*</sup> and [CMIP6](cmip6.whoi.edu) data servers | N/A (see instructions [below](#Accessing-the-climate-data-servers))
7/10/24 | Pre-processing using [```xarray```](https://docs.xarray.dev/en/stable/)  | [xarray_reference.ipynb](scripts/xarray_reference.ipynb)
7/11/24 | Defining a climate index | [woodshole_climate_index.ipynb](scripts/tutorials/woodshole_climate_index.ipynb)
7/16/24 | Climate model validation | [model_validation.ipynb](scripts/tutorials/model_validation.ipynb)
7/17/24 | Detecting climate change using models  | [climate_change_detection.ipynb](scripts/tutorials/climate_change_detection.ipynb)
7/18/24 | Model intercomparison using WHOI's servers  | [model_intercomparison.ipynb](scripts/tutorials/model_intercomparison.ipynb)  

__Note__: see [the task list](docs/task_list.md) for detailed instructions on how to run the tutorial notebooks from 7/11 - 7/18

<sup>*</sup>CMIP = Coupled Model Intercomparison Project   

## Description of other examples
Topic | Notebook
-- | --
Validating 2m-temperature and Gulf Stream position in CESM2 | [T2m_and_gulfstream_validation.ipynb](scripts/examples/T2m_and_gulfstream_validation.ipynb)
Detecting climate change in a stochastic model | [stochastic_large_ensemble.ipynb](scripts/examples/stochastic_large_ensemble.ipynb)
Reproducing results from [a recent Nature Geoscience paper](https://www.nature.com/articles/s41561-022-00971-w)<sup>1</sup>| [azores.ipynb](scripts/examples/azores.ipynb)

<sup>1</sup>Cresswell-Clay, N. et al. "Twentieth-century Azores High expansion unprecedented in the past 1,200 years". *Nat. Geosci.* 15, 548–553 (2022).

## Setup

### Getting the code
- Option 1: if you're comfortable with Github, fork [the repository](https://github.com/ktcarr/whoi-climate-data-tutorial/) (see [this page](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) for more on forking), then clone it to your PC.
- Option 2: Download the repository as a .zip file: go to [the repository home page](https://github.com/ktcarr/whoi-climate-data-tutorial/), then click "Code -> Download ZIP".

### Virtual environment
1. Set up mamba or conda (if not already). To set up, download and install miniforge following the instructions here: https://github.com/conda-forge/miniforge.
2. Navigate to the project home folder (e.g., with ```cd ~/whoi-climate-data-tutorial```)
3. Create a conda/mamba environment for the project with: ```mamba create -p ./envs``` and activate the environment with ```conda activate ./envs```
4. Next, install necessary packages in the environment with:<sup>3,4</sup>
    - (Mac/Linux) ```mamba env update -p ./envs --file environment.yml``` 
    - (Windows)  ```mamba env update -p ./envs --file environment_no_cdo.yml``` 
5. Install custom module (```src```) in the environment with ```pip install -e .```

<sup>3</sup>The CDO package, used for regridding data in this tutorial, is not available for Windows through conda (thanks to Haakon Pihlaja for catching this). This may cause the ```mamba env update``` command to "hang" when used with [environment.yml](environment.yml), the full list of packages (which includes CDO). While [it's possible to use CDO on Windows](https://code.mpimet.mpg.de/projects/cdo/wiki/Win32), it's probably not worth setting this up just for the tutorial. Instead, use the package list *without* CDO, [environment_no_cdo.yml](environment_no_cdo.yml). 

<sup>4</sup>If you're using conda and the ```conda install ...``` / ```conda env update ...``` commands are taking a long time, you could try [updating the solver to "libmamba"](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community). If this doesn't work, you could also try setting the channel priority to flexible, with ```conda config --set channel_priority flexible``` (thanks to Lilli Enders for suggesting this).

### Running the code (locally)
1. Navigate to project folder
2. Activate virtual environment (```conda activate ./envs```)
3. Start jupyter lab with by typing ```jupyter lab``` in terminal / command window

### Accessing the climate data servers
Note: to access the data, you must be on the WHOI network (i.e., on the WHOI wifi or connected by VPN).

#### Option 1 (preferred): mounting the network file system
- Windows and (non-Mac) Linux users: follow [online instructions for CMIP5](http://cmip5.whoi.edu/?page_id=40) or [for CMIP6](http://cmip6.whoi.edu/?page_id=50).
- Mac users: for CMIP6, open Finder, then select "Go" from the top menu bar and click "Connect to Server". Then, enter ```smb://vast.whoi.edu/proj/cmip6```. If prompted, enter your WHOI username (*without* "@whoi.edu") and password. Note the default mount location for the server is ```/Volumes/data```. For CMIP5, use the same process but with the following server address: ```smb://cmip5.whoi.edu```.

#### Option 2: downloading the data and running locally
__For reanalysis and model data used in the tutorial__:
- Download data from the [shared Google Drive folder](https://drive.google.com/drive/folders/1V-aHzoYYUrU6d5ExnxEORiZxeAAisx_e?usp=drive_link)

__For other (CMIP6) model output__:
- Go to [http://cmip6.whoi.edu/search](http://cmip6.whoi.edu/search).
- Click on the the "HTTP" link in the "Files" column for a dataset you'd like to download.
- On the next page, select individual files to download.
- (optional) Save these files to a folder called ```data``` in this project.

#### Option 3: running in the cloud
- Go to [https://colab.research.google.com/](https://colab.research.google.com/)
- In Google Colab, click "File -> Open notebook -> Upload" and select one of the tutorial notebooks. 
- Go to the [shared Google Drive folder](https://drive.google.com/drive/folders/1V-aHzoYYUrU6d5ExnxEORiZxeAAisx_e?usp=drive_link)
- Click the three dots next to the "climate-data" folder and click "Organize -> Add shortcut", then select "My Drive"
- Note: __if using Google Colab, the regridding components of the tutorial will not work__, owing to package compatibility issues (cannot import the ```xesmf``` package in Colab, possibly related to [this issue](https://github.com/conda-forge/esmf-feedstock/issues/91)).


## Description of high-level folders & files
Folder/file | Description
-- | --
```scripts/tutorials``` | contains jupyter notebooks used in tutorial
```scripts/examples``` | contains other examples of topics covered in the tutorial
```results``` | contains .md files with rendered output from tutorials and examples
```docs``` | contains FAQ and detailed instructions for completing tutorials
```src``` | custom module containing functions used in examples
```setup.py``` | file needed to import ```src``` module 
```environment*.yml``` | files containing list of packages needed for tutorial
```.gitignore``` | list of files and extensions ```git``` should ignore

## Other potentially useful links
- [The Good Research Code Handbook](https://goodresearch.dev/index.html) (A guide for how to organize research code for non-computer scientists)
- [Description of CMIP naming conventions, including variant ID](https://wcrp-cmip.org/cmip-data-access/)
- [CMIP6 overview paper](https://gmd.copernicus.org/articles/9/1937/2016/gmd-9-1937-2016.pdf)


