#!/usr/bin/env tcsh 

python fit_cogent_data.py --fit 4 --batch --sigma_n 5.0e-41 > log_5.0.log
python fit_cogent_data.py --fit 4 --batch --sigma_n 4.0e-41 > log_4.0.log
python fit_cogent_data.py --fit 4 --batch --sigma_n 3.0e-41 > log_3.0.log
python fit_cogent_data.py --fit 4 --batch --sigma_n 2.0e-41 > log_2.0.log
python fit_cogent_data.py --fit 4 --batch --sigma_n 1.0e-41 > log_1.0.log
python fit_cogent_data.py --fit 4 --batch --sigma_n 0.8e-41 > log_0.8.log
python fit_cogent_data.py --fit 4 --batch --sigma_n 0.6e-41 > log_0.6.log
python fit_cogent_data.py --fit 4 --batch --sigma_n 0.4e-41 > log_0.4.log
python fit_cogent_data.py --fit 4 --batch --sigma_n 0.2e-41 > log_0.2.log
python fit_cogent_data.py --fit 4 --batch --sigma_n 0.1e-41 > log_0.1.log
