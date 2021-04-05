# RODAN
A fully convolutional architecture for basecalling nanopore RNA sequencing data

Source code for paper LINK

Generated Taiyaki RNA data: https://doi.org/10.5281/zenodo.4556884

RNA training and validation data: https://doi.org/10.5281/zenodo.4556950

RNA test data: https://doi.org/10.5281/zenodo.4557004



## Requirements
* Python 3
* torch >= 1.2 <= 1.5.1
* numpy
* h5py
* ont-fast5-api
* fast-ctc-decode
* pytorch_ranger (only for training)

To basecall:

`basecall.py /path/to/fast5files > outfile.fasta`

To train:

`mkdir runs`

`model.py -c rna.config -n NAME -l`

### Parameters
-l for label smoothing

### Test data
Five samples of human RNA fast5 data is provided in test-data.tgz.

### Memory errors
If you run out of memory, reduce the batch size with the basecaller with "-b 100" or lower. The default is 200.
