chelsa_cmip6
-----------
This package contains functions to creates monthly high-resolution 
climatologies for min-, max-, and mean temperature, precipitation rate 
and bioclimatic variables from anomalies and using CHELSA V2.1 as 
baseline high resolution climatology. Only works for GCMs for
hich tas, tasmax, tasmin, and pr are available. It is part of the
CHELSA Project: (CHELSA, <https://www.chelsa-climate.org/>).




COPYRIGHT
---------
(C) 2021 Dirk Nikolaus Karger



LICENSE
-------
chelsa_cmip6 is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the
Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

chelsa_cmip6 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with chelsa_cmip6. If not, see <http://www.gnu.org/licenses/>.



REQUIREMENTS
------------
chelsa_cmip6 is written in Python 3. It has been tested to run well with the
following Python release and package versions.
- python 3.6.5 
- xarray 0.16.2
- requests 2.25.1
- numpy 1.19.5
- rasterio 1.2.1
- pandas 1.1.5
- zarr 2.6.1
- gcsfs 0.7.2
- datetime 3.9.2
- scipy 0.19.1



HOW TO USE
----------
The chelsa_cmip6 module provides functions to create monthly climatologies from climate
simulation data from CMIP6 using climate observation data from CHELSA V.2.1
at a 0.0083333° grid resolution for a given area of choice.

The GetClim module contains classes and functions to connect to CMIP6 data
via the Google cloud storage and to read the data into xarrays. It also creates
monthly climatologies using the delta change anomally correction method for a given 
time period. 

The BioClim module contains classes calculating various bioclimatic parameters
from climatological data (see: https://chelsa-climate.org/bioclim).

The delta change method that is applied is relatively insensitive towards individual model 
bias of a GCM, as it only uses the difference (ratio) for a given variable between
a reference period and a future period. In case of temperature an additive delta change 
is applied. In case of precipitation a multiplicative delta change is applied by 
adding a constant of 0.0000001 kg m^-2 s^⁻1 to both the reference and the future data
to avoid division by zero. 

The code only runs for CMIP6 models for which all needed variables tas, tasmax, tasmin, pr,
are available for both the reference and the future period.

The standard reference period is 1981-01-01 - 2010-12-31. If another reference period is 
chosen, the code conducts a delta change for this period as well. Best practice would be to 
choose the standard reference period.

CITATION:
------------
If you need a citation for the output, please refer to the article describing the high
resolution climatologies:

Karger, D.N., Conrad, O., Böhner, J., Kawohl, T., Kreft, H., Soria-Auza, R.W., Zimmermann, N.E., Linder, P., Kessler, M. (2017). Climatologies at high resolution for the Earth land surface areas. Scientific Data. 4 170122. https://doi.org/10.1038/sdata.2017.122


EXAMPLE: 
------------
You can use the program by running the following command in the terminal:

if you are interested in future climate data, you can run the function for example like this:


python chelsa_cmip6.py --activity_id 'ScenarioMIP' --table_id 'Amon' --experiment_id 'ssp585' --institution_id 'MPI-M' --source_id 'MPI-ESM1-2-LR' --member_id 'r1i1p1f1' --refps '1981-01-15' --refpe '2010-12-15' --fefps '2041-01-15' --fefpe '2070-12-15' --xmin 5.3 --xmax 10.4 --ymin 46.0 --ymax 47.5 --output '/home/karger/scratch/'


important is that the combination of activity_id 'ScenarioMIP' and e.g. experiment_id 'ssp585' is set to a combination that exists.
You can also get historical data but in that case, activity_ID, experiment_id, and fefps and fefps need to be changed. E.g. 


python chelsa_cmip6.py --activity_id 'CMIP' --table_id 'Amon' --experiment_id 'historical' --institution_id 'MPI-M' --source_id 'MPI-ESM1-2-LR' --member_id 'r1i1p1f1' --refps '1981-01-15' --refpe '2010-12-15' --fefps '1851-01-15' --fefpe '1880-12-15' --xmin 5.3 --xmax 10.4 --ymin 46.0 --ymax 47.5 --output '/home/karger/scratch/'


it is important that your fefps and fefpe are covered by the experiment_id and activity_id.


These reference periods are possible for example:


'ScenarioMIP' - 2016-01-01 - 2100-12-31


'CMIP' - 1850-01-01 - 2015-12-31


refps and refpe need to be in the range 1850-01-01 - 2015-12-31.


or within python by importing the chelsa_cmip6 function:
from classes.GetClim import chelsa_cmip6


SINGULARITY
------------
All dependencies are also resolved in the singularity container '/singularity/chelsa_cmip6.sif'. Singularity needs to be installed on the respective linux system you are using. 
An installation guide can be found here: https://sylabs.io/guides/3.3/user-guide/quick_start.html#quick-installation-steps

If you use chelsa_cmip6 together with singularity the command should be slightly modified:
singularity exec /singularity/chelsa_cmip6.sif python3 chelsa_cmip6.py --activity_id 'CMIP' --table_id 'Amon' --experiment_id 'historical' --institution_id 'MPI-M' --source_id 'MPI-ESM1-2-LR' --member_id 'r1i1p1f1' --refps '1981-01-15' --refpe '2010-12-15' --fefps '1851-01-15' --fefpe '1880-12-15' --xmin 5.3 --xmax 10.4 --ymin 46.0 --ymax 47.5 --output '/home/karger/scratch/'

tested with singularity version 3.3.0-809.g78ec427cc
but newer versions usually work as well.


CHECKING IF ALL NEEDED INPUT IS AVAILABLE
------------
Not all models and activities provied all the neccessary input needed for chelsa_cmip6.py.
chelsa_cmip6.py will only work for GCMs that are both available for the historical period
and the respective scenario of interest. You can check this by using the CMIP6 data search
interface on e.g. https://esgf-node.llnl.gov/search/cmip6/ 
There you can filter for the different parameters (e.g. experiment_id) and see if a dataset
exists. E.g. by using the parameters given in the example. To check if also the historical
data exists for the model, just change the activity_id to 'CMIP' and the experiment_id to 'historical'.
Make sure the four variables needed do exist both for the scenario and the historical period:

These variables are needed:
- pr
- tas
- tasmax
- tasmin

OUTPUT
------------
The output consist of netCDF4 files. There will be different files for each variable and seperatly for
the reference (refps - refpe) and the future period (fefps - fefpe). 
Additionally, there will be netCDF4 files for the 
different bioclimatic variables each for both the reference (refps - refpe) and the future period (fefps - fefpe). 


CONTACT
-------
<dirk.karger@wsl.ch>



AUTHOR
------
Dirk Nikolaus Karger
Swiss Federal Research Institute WSL
Zürcherstrasse 111
8903 Birmensdorf 
Switzerland
