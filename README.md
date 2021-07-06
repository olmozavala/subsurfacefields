#Folders

Profiles: `/data/COAPS_nexsan/people/xbxu/hycom/GLBb0.08/profile`

 INFO:

`test_Y2001.nc` has the profiles that irregularly distributed throughout each days in the year …   <br>
`test_Y2001_05deg.nc` has the profiles that are sa**mpled in every 1/2 degree, every 10 days ...

#SRC Files

## 1_AnalizeData (geoeoas)
It contains multiple functions for the analysis of the source/model data. <br>
`img_generation_3D` It makes plots of SST, SSH, SSS for the *new* type of 3D files (not profiles) <br>
`img_generation_all` Makes profiles and maps from the original files. To make maps of the locations you 
can use `plotSparseDataFiles` and for making random profile plots you use `plot5DegDataFiles`.

## 6_Dash_Error_Stats (AIEOAS)
This program reads all the networks stored in the `summary.csv` file and generates 
a numpy file `nn_prediction.npy` at the **output** prediction folder of the network with 
all the predictions of each newtork. 

## 7_Dash_Error_Stats (AIEOAS)
This program creates the web dashboards from the `summary.csv` file. Depending on the
selected network it will search for the corresponding `nn_prediction.npy` file and make the plots.


