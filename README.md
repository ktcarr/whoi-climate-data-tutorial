# CMIP5/6 Tutorial

## Topics to be covered
- model validation
- external forcing vs. internal variability
- intermodel spread vs. internal variability
- confidence intervals?

## General outline

0. (optional) Setting up a virtual environment and version control.

1. Accessing CMIP data at WHOI.

1.5 Loading/plotting data

2. Model validation 
    - what metrics to use?
    - comparison to observations/reanalysis: biases in mean/variability?

3. Computing ensemble statistics (e.g., mean and range)
  

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