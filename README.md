This Repository Contains Benchmark Results for Brainome's Table Compiler

Instructions:
The script validate.sh requires Python3, wget and a c-shell. It will download each dataset directly from OpenML and run the compiled predictor on it for validation.

Each predictor requires only the Python standard library and some of them require numpy.
Numpy is available here: https://numpy.org/

The predictors can also be invoked directly using

python3 Predictors/dataset.py Data/dataset.csv -validate

where dataset is the name of a dataset.

We also invite you to take a look at the source code of the preditors.

report.txt shows our results compared to what was reported on OpenML.org (as of March 24th, 2020).
hardware.txt shows CPU and GPU configuration used to obtain the result. We used a single desktop machine. 
