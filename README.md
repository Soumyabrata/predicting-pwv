# Predicting GPS-based PWV Measurements Using Exponential Smoothing
With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:
> S. Manandhar&ast;, S. Dev&ast;, Y. H. Lee and S. Winkler, Predicting GPS-based PWV Measurements Using Exponential Smoothing, IEEE AP-S Symposium on Antennas and Propagation and USNC-URSI Radio Science Meeting, 2019 (&ast; Authors contributed equally).

![summary](./results/aps2019asummary.png)

Please cite the above paper if you intend to use whole/part of the code. This code is only for academic and research purposes.

## Code organization
The codes are written in `python`. The codes are tested in `python3` version.

### Dependencies

+ matplotlib: `pip3 install matplotlib`
+ Tkinter: `sudo apt-get install python3-tk`
+ statsmodels: `pip3 install statsmodels`
+ scipy: `pip3 install scipy`

## Usage

1. `python3 pwv_forecasting.py`: Run this script to compute the predicted PWV values, based on the historical PWV data. This generated Figure 1 of the paper. You need to change the value of the parameter `start_index` in the file, to check for different slices of time-series data.
2. `python3 grid_search.py`: Run this script to obtain the distribution of RMSE values w.r.t. historical data and lead times. This generates Figure 2 of the paper. It saves the figure as `./results/rmse.pdf` and the corresponding numpy array as `rmse_matrix_for_grid.npy`. 
3. `python3 benchmarking.py`: Run this script to obtain the benchmarking results of the different methods. This generates the results in Table 1 of the paper. The results are stored in `./results/comparison.txt` file.


## Scripts
+ `benchmarking.py`: For a particular past time observation, we compare the performance of various methods for various lead times, and store in a text file.
+ `grid_search.py`: Performs the experiments for various lead times and various historical observations. This is repeated for several times, and the average is reported.
+ `pwv_forecasting.py`: Performs pwv forecasting for sample datapoints and plots the figure with train and test legends
+ `read_matfile.py`: Reads the matlab files, performs pre-processing and returns the data series
