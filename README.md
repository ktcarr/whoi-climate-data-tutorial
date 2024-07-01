# CMIP5/6 Tutorial

## Set up
#### Virtual environment
1. Set up mamba or conda (if not already). To set up, download and install miniforge following the instructions here: https://github.com/conda-forge/miniforge.
2. Create a project folder; e.g., with ```mkdir ~/cmip_tutorial``` and navigate to the project with ```cd ~/cmip_tutorial```
3. Create a conda/mamba environment for the project with: ```mamba create -p ./envs``` and activate the environment with ```conda activate ./envs```
4. Install necessary packages in the environment with ```mamba env update -p ./envs --file environment.yml```

#### Accessing the climate data servers
- Windows and (non-Mac) Linux users: follow [online instructions for CMIP5](http://cmip5.whoi.edu/?page_id=40) or [for CMIP6](http://cmip6.whoi.edu/?page_id=50).
- Mac users: for CMIP6, open Finder, then select "Go" from the top menu bar and click "Connect to Server". Then, enter ```smb://vast.whoi.edu/proj/cmip6```. If prompted, enter your WHOI username (*without* "@whoi.edu") and password. Note the default mount location for the server is ```/Volumes/data```. For CMIP5, use the same process but with the following server address: ```smb://cmip5.whoi.edu```.

## Topics to be covered
- model validation
- external forcing vs. internal variability
- intermodel spread vs. internal variability
- confidence intervals?
- removing trend/seasonal cycle
- apply ufunc
- chunking?
- regridding
- area-weighted mean
- first section on 'reducing' data? (could have CDO as well)

## General outline
  

## Azores High example
#### References
- Cresswell-Clay, N. et al. Twentieth-century Azores High expansion unprecedented in the past 1,200 years. Nat. Geosci. 15, 548–553 (2022).  
- Thatcher, D. L. et al. Iberian hydroclimate variability and the Azores High during the last 1200 years: evidence from proxy records and climate model simulations. Clim Dyn 60, 2365–2387 (2023).


#### Datasets and metrics
- (observations/reanalysis): OISST, HadISST, ERSST. Choose one from the options located here: ```/vortexfs1/share/clidex/data/obs/SST```. Actually, these are not on CMIP6 server; might be best to fall back on ERA5.  
- (model): TBD  
    
#### Metrics
- location of maximum meridional gradient  
- location of specified isotherm (see Andres, 2016)  
- estimate subtropical gyre's center of mass (see Yang et al., 2020)  

## Gulf Stream example
#### References
- Todd, R. E. & Ren, A. S. Warming and lateral shift of the Gulf Stream from in situ observations since 2001. Nat. Clim. Chang. 13, 1348–1352 (2023).  
- Yang, H. et al. Poleward Shift of the Major Ocean Gyres Detected in a Warming Climate. Geophysical Research Letters 47, (2020).  
- Andres, M. On the recent destabilization of the Gulf Stream path downstream of Cape Hatteras. Geophysical Research Letters 43, 9836–9842 (2016).

#### 1. Datasets and metrics
- (observations/reanalysis): OISST, HadISST, ERSST. Choose one from the options located here: /vortexfs1/share/clidex/data/obs/SST. Actually, these are not on CMIP6 server; might be best to fall back on ERA5.  
- (model): TBD  
    
#### 2. Metrics
- location of maximum meridional gradient  
- location of specified isotherm (see Andres, 2016)  
- estimate subtropical gyre's center of mass (see Yang et al., 2020)

## polar/biological/paleo example?


# Scratch

## Other project ideas

#### AMOC decline 
- intermodel spread  
- emergent constraint: current strength vs. projected change  
- references: Weijer et al. (2020) and Bellomo et al. (2021)

#### Wet-get-wetter / dry-get-drier
- is delta(T) a good predictor of delta(P)? Over land vs. ocean?
- references: Held and Soden (2006) and O'Gorman and Schneider (2009)

## Other topic ideas
- Emergent constraints (e.g., current AMOC strength vs. change, surface T vs. P, surface winds vs. GS strength)
- EOFs (too complicated?)

# To run on Climex:
- ```. ~/mamba.sh```
- ```mamba activate ./envs```
- ```jupyter lab --no-browser```
- (on laptop) ```ssh -N -f -L localhost:8889:localhost:8888 kcarr@climex.whoi.edu```
