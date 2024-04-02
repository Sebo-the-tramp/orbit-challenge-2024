#
# Created on Sat Feb 17 2024
#
# The MIT License (MIT)
# Copyright (c) 2024 Sebastian Cavada
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


# This code converts the annotations from the orbit dataset to a parquet file for faster access

import glob
import pandas as pd

try:
    import fastparquet
except ImportError:
    print("I am saving you time, instead of waiting for the error at the end, install pyarrow -> pip install fastparquet")

def get_dataframefiles(json_files):
    return [pd.read_json(file).astype(bool).transpose() for file in json_files]


# read all files in fast way
train_files = glob.glob('./train/*')
print("Creating Train dataframe...")
train_df = pd.concat(get_dataframefiles(train_files), ignore_index=False)
print("Train dataframe created")
print("Saving train to parquet files")
train_df.to_parquet('train.parquet')


test_files = glob.glob('./test/*')
print("Creating Test dataframe...")
test_df = pd.concat(get_dataframefiles(test_files), ignore_index=False)
print("Test dataframe created")
print("Saving test to parquet files")
test_df.to_parquet('test.parquet')


val_files = glob.glob('./validation/*')
print("Creating Val dataframe...")
val_df = pd.concat(get_dataframefiles(val_files), ignore_index=False)
print("Val dataframe created")
print("Saving val to parquet files")
val_df.to_parquet('val.parquet')