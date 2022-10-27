# RODAN
A fully convolutional architecture for basecalling nanopore RNA sequencing data

Generated Taiyaki RNA data: https://doi.org/10.5281/zenodo.4556884

RNA training and validation data: https://doi.org/10.5281/zenodo.4556950

RNA test data: https://doi.org/10.5281/zenodo.4557004

## Requirements
* Python 3
* torch >= 1.4.0 <= 1.8.0
* numpy
* h5py
* ont-fast5-api
* fast-ctc-decode
* pyyaml
* tensorboard
* pytorch-ranger (only for training)

## Installation

Create a python virtual environment. 
```
python3 -m venv virtualenv
source virtualenv/bin/activate
git clone https://github.com/biodlab/RODAN.git
cd RODAN
pip install -r requirements.txt
```

## Basecalling

To basecall (must be run from root directory):

`./basecall.py /path/to/fast5files > outfile.fasta`

Basecall will recursively search in the specified directory for all fast5 files which can be single or multi fast5 files.

## Training

To train, download the RNA training data from the above link.

```
mkdir runs
pip install pytorch-ranger
./model.py -c rna.config -n NAME -l
```

### Parameters
-c for configuration file\
-l for label smoothing\
-n the name for the run, the model weights, configuration, and results will be saved in the runs directory\
-v verbose

### Test data
Five samples of human RNA fast5 data is provided in test-data.tgz.

### Memory errors
If you run out of memory, reduce the batch size with the basecaller with "-b 100" or lower. The default is 200.

### License
MIT License.
