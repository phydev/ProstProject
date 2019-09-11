# ProstProject
## Summer School in Computational Biology @ UC 

*Modelling prostate cancer growth using a phase-field model.*

**Research team**: Oliver, Hygor, Diana, Carlos, Milan, Mauricio & Rui.

## Files:
- **src/**  ~ *source files*
  - ProstProject.py  ~ *full numpy/scipy implementation*
  - src/loopimp.py  ~ *explicit loop implementation*
- **notebooks/** ~ *jupyter notebooks*
  - compare_speed.ipynb ~ *comparing the computational time between methods*
  - parvariation.ipynb ~ *implementation with OpenCV convolution *
  - simulation_init_par.ipynb ~ *simulation with the appropriate set of parameters*
- **data/** ~ *graphs & data files*
  - result_comp.png ~ *accuracy comparison between the convolution method & explicit loops*
  - speed_comp.png ~ *computational efficiency*

## Logbook

### 1st day:
- Laplace Operator;
- Euler's Method for time integration;
- Presented and implemented the phase-field basic model.

### 2nd day:
- Implemented the complete model including the nutrient field.

### 3rd day:
- Meeting for discussions about the preliminary results.

### 4th day:
- Search for the right set of parameters in order to reproduce the paper results.


## Results
<img src="https://phydev.github.io/ProstProject/data/result_comp.png" width="400" height="400">
<img src="https://phydev.github.io/ProstProject/data/speed_comp.png" width="400" height="400">

