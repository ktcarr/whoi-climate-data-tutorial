# Climate data analysis @ WHOI: a tutorial
The purpose of this tutorial is to provide practical guidance on how to analyze gridded climate data stored on WHOI's servers using python. The tutorial is designed to take place over $\sim$6 hour-long sessions, and is split into two parts. In the first part, we'll step through a template for assessing climate change: (i) defining a climate index, (ii) evaluating a model's ability to represent processes which influence this index, and (iii) assessing long-term changes in the index by comparing a model's historical and pre-industrial control simulations. In the second part, we'll see how these principles are applied in state-of-the-art research by reproducing results from a recent study of the Azores High$^1$.

$^1$Cresswell-Clay, N. et al. "Twentieth-century Azores High expansion unprecedented in the past 1,200 years". *Nat. Geosci.* 15, 548–553 (2022).

## Outline
Topic | Notebook
-- | --
Connecting to the [CMIP5](cmip5.whoi.edu)$^\dagger$ and [CMIP6](cmip6.whoi.edu) data servers | N/A (see instructions [below](#Accessing-the-climate-data-servers))
Pre-processing using [```xarray```](https://docs.xarray.dev/en/stable/) and defining a climate index | [0_xarray_tutorial.ipynb](scripts/0_xarray_tutorial.ipynb)
Climate model validation | [1_model_validation_tutorial.ipynb](scripts/1_model_validation_tutorial.ipynb)
Asessing climate change using models | [2_cmip_tutorial.ipynb](scripts/2_cmip_tutorial.ipynb)
Azores High example (1/2) | [azores_tutorial.ipynb](scripts/azores_tutorial.ipynb)
Azores High example (2/2) | [azores_tutorial.ipynb](scripts/azores_tutorial.ipynb)


$^\dagger$CMIP = Coupled Model Intercomparison Project   

## Set up
### Virtual environment
1. Set up mamba or conda (if not already). To set up, download and install miniforge following the instructions here: https://github.com/conda-forge/miniforge.
2. Create a project folder; e.g., with ```mkdir ~/cmip_tutorial``` and navigate to the project with ```cd ~/cmip_tutorial```
3. Create a conda/mamba environment for the project with: ```mamba create -p ./envs``` and activate the environment with ```conda activate ./envs```
4. Install necessary packages in the environment with ```mamba env update -p ./envs --file environment.yml```
5. Install custom module (```src```) in the environment with ```pip install -e .```

For guidance on how to structure your code, I highly recommend [The Good Research Code Handbook](https://goodresearch.dev/index.html).

### Accessing the climate data servers
Note: to access the data, you must be on the WHOI network (i.e., on the WHOI wifi or connected by VPN).

#### Option 1 (preferred): mounting the network file system
- Windows and (non-Mac) Linux users: follow [online instructions for CMIP5](http://cmip5.whoi.edu/?page_id=40) or [for CMIP6](http://cmip6.whoi.edu/?page_id=50).
- Mac users: for CMIP6, open Finder, then select "Go" from the top menu bar and click "Connect to Server". Then, enter ```smb://vast.whoi.edu/proj/cmip6```. If prompted, enter your WHOI username (*without* "@whoi.edu") and password. Note the default mount location for the server is ```/Volumes/data```. For CMIP5, use the same process but with the following server address: ```smb://cmip5.whoi.edu```.

#### Option 2: downloading the data and running locally
__For ERA5 reanalysis__:
- Download SST or $T_{2m}$ data from [shared Google Drive folder](https://drive.google.com/drive/folders/1FQBVTQWpvVPIrHFYlZc_OLl93JrLOWze?usp=drive_link)

__For CMIP6 model output__:
- Go to [http://cmip6.whoi.edu/search](http://cmip6.whoi.edu/search).
- Click on the the "HTTP" link in the "Files" column for a dataset you'd like to download.
- On the next page, select individual files to download.
- (optional) Save these files to a folder called ```data``` in this project.

#### Option 3: running in the cloud
- Save the tutorial notebooks to Google Drive
- Go to [https://colab.research.google.com/](https://colab.research.google.com/)
- In Google Colab, click "File -> Open notebook -> Google Drive" and select one of the tutorial notebooks. 
- Go to the [shared Google Drive folder](https://drive.google.com/drive/folders/1V-aHzoYYUrU6d5ExnxEORiZxeAAisx_e?usp=drive_link)
- Click the three dots next to the "climate-data" folder and click "Organize -> Add shortcut", then select "My Drive"


## Description of high-level folders & files:
Folder/file | Description
-- | --
```scripts``` | contains ```.ipynb``` notebooks used in tutorial
```src``` | custom module containing functions used in the tutorials
```setup.py``` | file needed to import ```src``` module 
```environment.yml``` | list of packages needed for tutorial
```.gitignore``` | list of files and extensions ```git``` should ignore