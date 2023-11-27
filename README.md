# Starting kit on the bike counters dataset

![GH Actions](https://github.com/ramp-kits/bike_counters/actions/workflows/main.yml/badge.svg)

## Getting started

### Download the data,

Download the data files,
 - [train.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/train.parquet)
 - [test.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/test.parquet)

and put them into into the data folder.

Note that the `test.parquet` file is different from the actual `final_test.parquet` used for the evaluation on Kaggle. This file is just here for convenience.

### Install

To run the notebook you will need the dependencies listed
in `requirements.txt`. 

It is recommended to create a new virtual environement for this project. For instance, with conda,
```bash
conda create -n bikes-count python=3.10
conda activate bikes-count
```

You can install the dependencies with the following command-line:

```bash
pip install -U -r requirements.txt
```


### Challenge description

Get started on this challenge with the
[dedicated notebook](bike_counters_starting_kit.ipynb).

Launch the notebook using:

```bash
jupyter lab ./bike_counters_starting_kit.ipynb
```

### Submissions

Upload your script file `.py` to Kaggle using the Kaggle interface directly.
The platform will then execute your code to generate your submission csv file, and compute your score.
Note that your submission .csv file must have the columns "Id" and "bike_log_count", and be of the same length as `final_test.parquet`.
