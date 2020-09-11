This Repository Contains Benchmark Results for Brainome's Table Compiler

The datasets where chosen to be able to compare against the Capital One/UIUC AutoML benchmark published here:
https://arxiv.org/pdf/1908.05557.pdf

Instructions:
The script get-data requires Python3, wget and a c-shell. It will read files.csv and download each dataset directly from OpenML. To validate all the predictor, use the command python3 _validate.py.

Each predictor requires only the Python standard library and some of them require numpy.
Numpy is available here: https://numpy.org/

The predictors can also be invoked directly using

python3 Predictors/*dataset.py Data/dataset.csv -validate

where dataset is the name of a dataset.

We also invite you to take a look at the source code of the preditors.

binary_report.txt shows our results compared to what was reported on OpenML.org (as of March 24th, 2020). None of the binary classifiers overfit.
multiclass_report.txt shows our results compared to what was reported on OpenML.org (as of May 15th, 2020).
hardware.txt shows CPU and GPU configuration used to obtain the result. We used a single desktop machine. 
