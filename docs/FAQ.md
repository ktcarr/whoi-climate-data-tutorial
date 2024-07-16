## How to resolve ```NetCDF: HDF``` error?
(typically seen when trying to open data on the CMIP6 server using xarray or NetCDF packages)

### On Mac
Add the following line to your .bashrc, .bash_profile, or .zshrc file:<sup>1</sup>
```bash
export HDF5_USE_FILE_LOCKING=FALSE
```
<sup>1</sup>These files should be located in your home folder (you can navigate to your home folder by entering ```cd ~``` in the terminal).

### On Windows (untested solution)
*(Note: this solution was suggested by IS at WHOI but hasn't been tested yet)*. On Windows, we're going to do the same thing as on Mac (set the ```HDF5_USE_FILE_LOCKING``` variable to ```FALSE```), but the process for setting environment variables is different. [This page from Microsoft](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-7.4#set-environment-variables-in-the-system-control-panel) describes how to set the environment variables ([this blog post](https://www.architectryan.com/2018/08/31/how-to-change-environment-variables-on-windows-10/) shows a more detailed example). To try this potential solution, follow the instructions at the links above for setting an environment variable, and add a new variable called ```HDF5_USE_FILE_LOCKING``` with a value of ```FALSE```.

__Note__: for both solutions, you may need to restart your shell/terminal and Jupyter session for changes to take effect. 

