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



# James Notes:

to satisfy torch versions, I think I need python 3.8 (was trying to use python3.10)
So using deadsnakes
```
sudo apt install python3.8 python3.8-dev python3.8-venv
```

Yep, that seems to work!

getting torch 1.8.0

Okay all done. Quick `pip install pyslow5` to install that for slow5 changes.

`./basecall.py test/ > outfile.fasta`

so first try got an error:

```
(virtualenv) jamfer@garvan-work:~/Dropbox/Bioinformatics/tools/repos/RODAN$ ./basecall.py test/ > outfile.fasta
Traceback (most recent call last):
  File "./basecall.py", line 17, in <module>
    import model as network
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/model.py", line 12, in <module>
    from torch.utils.tensorboard import SummaryWriter
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/torch/utils/tensorboard/__init__.py", line 8, in <module>
    from .writer import FileWriter, SummaryWriter  # noqa F401
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/torch/utils/tensorboard/writer.py", line 9, in <module>
    from tensorboard.compat.proto.event_pb2 import SessionLog
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/tensorboard/compat/proto/event_pb2.py", line 17, in <module>
    from tensorboard.compat.proto import summary_pb2 as tensorboard_dot_compat_dot_proto_dot_summary__pb2
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/tensorboard/compat/proto/summary_pb2.py", line 16, in <module>
    from tensorboard.compat.proto import tensor_pb2 as tensorboard_dot_compat_dot_proto_dot_tensor__pb2
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/tensorboard/compat/proto/tensor_pb2.py", line 16, in <module>
    from tensorboard.compat.proto import resource_handle_pb2 as tensorboard_dot_compat_dot_proto_dot_resource__handle__pb2
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/tensorboard/compat/proto/resource_handle_pb2.py", line 16, in <module>
    from tensorboard.compat.proto import tensor_shape_pb2 as tensorboard_dot_compat_dot_proto_dot_tensor__shape__pb2
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/tensorboard/compat/proto/tensor_shape_pb2.py", line 36, in <module>
    _descriptor.FieldDescriptor(
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/google/protobuf/descriptor.py", line 560, in __new__
    _message.Message._CheckCalledFromGeneratedFile()
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
```

`pip freeze`:

```
absl-py==1.3.0
fast_ctc_decode==0.3.2
future==0.18.2
grpcio==1.50.0
h5py==3.7.0
importlib-metadata==5.0.0
Markdown==3.4.1
MarkupSafe==2.1.1
numpy==1.23.4
ont-fast5-api==4.1.0
packaging==21.3
progressbar33==2.4
protobuf==4.21.8           <========== this one
pyparsing==3.0.9
pyslow5==0.8.0
PyYAML==6.0
six==1.16.0
tensorboard==1.15.0
torch==1.8.0
typing_extensions==4.4.0
Werkzeug==2.2.2
zipp==3.10.0
```

So downgrading protobuf

`pip uninstall protobuf`
`pip install protobuf==3.20`


trying again


`./basecall.py test/ > outfile.fasta`

Seems my laptop GPU isn't going to work. Time to move to server.

```
/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/torch/cuda/__init__.py:104: UserWarning: 
NVIDIA GeForce RTX 3050 Ti Laptop GPU with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
If you want to use the NVIDIA GeForce RTX 3050 Ti Laptop GPU GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))
Process Process-2:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "./basecall.py", line 152, in mp_gpu
    out=model.forward(event)
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/model.py", line 247, in forward
    x = self.convlayers(x)
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/torch/nn/modules/container.py", line 119, in forward
    input = module(input)
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/model.py", line 165, in forward
    x = self.act1(x)
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/virtualenv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/jamfer/Dropbox/Bioinformatics/tools/repos/RODAN/model.py", line 73, in forward
    return x *( torch.tanh(torch.nn.functional.softplus(x)))
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

okay now on server with Tesla V100 cards

re-install as above

```
tar -zxf test-data.tgz
mkdir test
mv *.fast5 test/
```

`./basecall.py test/ > outfile.fasta`

and it works!

now for slow5, need to convert fast5 files to blow5, but first need to merge singles into multi fast5

```
mkdir multi
single_to_multi_fast5 -i test/ -s multi/
```

then from multi to blow5

```
mkdir slow5

slow5tools f2s -d slow5/ multi/

# but it throws an error cause these are not normal fast5 reads...
[search_and_warn::WARNING] slow5tools-v0.6.0: Weird or ancient fast5: converting the attribute Raw/read_number from H5T_STD_U32LE to SLOW5_INT32_T for consitency.
[search_and_warn::WARNING] slow5tools-v0.6.0: Weird or ancient fast5: converting the attribute Raw/start_mux from H5T_STD_U32LE to SLOW5_UINT8_T for consitency.
[search_and_warn::WARNING] slow5tools-v0.6.0: Weird or ancient fast5: converting the attribute Raw/read_number from H5T_STD_U32LE to SLOW5_INT32_T for consitency. This warning is suppressed now onwards.
[search_and_warn::WARNING] slow5tools-v0.6.0: Weird or ancient fast5: converting the attribute Raw/start_mux from H5T_STD_U32LE to SLOW5_UINT8_T for consitency. This warning is suppressed now onwards.
[fast5_attribute_itr::ERROR] Ancient fast5: Different run_ids found in an individual multi-fast5 file. Cannot create a single header slow5/blow5. Consider --allow option.
[read_fast5::ERROR] Bad fast5: Could not iterate over the read groups in the fast5 file multi//batch_0.fast5.
[f2s_child_worker::ERROR] Bad fast5: Could not read contents of the fast5 file 'multi//batch_0.fast5'.


# so let's add --allow to get around that

slow5tools f2s --allow -d slow5/ multi/
```

and now we have our blow5 file

`./basecall.py slow5/batch_0.blow5 > outfile_slow5.fasta`

and that works and the reads are the same!

winning!