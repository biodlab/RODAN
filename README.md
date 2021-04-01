# RODAN
A fully convolutional architecture for basecalling nanopore RNA sequencing data

Source code for paper LINK

Generated Taiyaki RNA data: https://doi.org/10.5281/zenodo.4556884

RNA training and validation data: https://doi.org/10.5281/zenodo.4556950

RNA test data: https://doi.org/10.5281/zenodo.4557004



## Requirements
* Python 3
* pytorch >= 1.2 <= 1.5.1
* numpy
* h5py
* ont-fast5-api
* fast-ctc-decode

To basecall:

`basecall.py /path/to/fast5files > outfile.fasta`

To train:

`mkdir runs

model.py -c rna.config -n NAME -l`

### Parameters
-l for label smoothing

