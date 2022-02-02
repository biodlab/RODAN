#!/usr/bin/env python
# 
# RODAN
# v1.0
# (c) 2020,2021,2022 Don Neumann
#

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import h5py
from torch.autograd import Variable
import sys, argparse, shutil, pickle, os, copy, math
from collections import OrderedDict
import ont

defaultconfig = {"name":"default", "seqlen":4096, "epochs":30, "optimizer":"ranger", "lr":2e-3, "weightdecay":0.01, "batchsize":30, "dropout": 0.1, "activation":"mish", "sqex_activation":"mish", "sqex_reduction":32, "trainfile":"rna-train.hdf5", "validfile":"rna-valid.hdf5", "amp":False, "scheduler":"reducelronplateau", "scheduler_patience":1, "scheduler_factor":0.5, "scheduler_threshold":0.1, "scheduler_minlr": 1e-05, "scheduler_reduce":2, "gradclip":0, "train_loopcount": 1000000, "valid_loopcount": 1000, "tensorboard":False, "saveinit":False,
        "vocab": [ '<PAD>', 'A', 'C', 'G', 'T' ]}

rna_default = [[-1, 256, 0, 3, 1, 1, 0], [-1, 256, 1, 10, 1, 1, 1], [-1, 256, 1, 10, 10, 1, 1], [-1, 320, 1, 10, 1, 1, 1], [-1, 384, 1, 15, 1, 1, 1], [-1, 448, 1, 20, 1, 1, 1], [-1, 512, 1, 25, 1, 1, 1], [-1, 512, 1, 30, 1, 1, 1], [-1, 512, 1, 35, 1, 1, 1], [-1, 512, 1, 40, 1, 1, 1], [-1, 512, 1, 45, 1, 1, 1], [-1, 512, 1, 50, 1, 1, 1], [-1, 768, 1, 55, 1, 1, 1], [-1, 768, 1, 60, 1, 1, 1], [-1, 768, 1, 65, 1, 1, 1], [-1, 768, 1, 70, 1, 1, 1], [-1, 768, 1, 75, 1, 1, 1], [-1, 768, 1, 80, 1, 1, 1], [-1, 768, 1, 85, 1, 1, 1], [-1, 768, 1, 90, 1, 1, 1], [-1, 768, 1, 95, 1, 1, 1], [-1, 768, 1, 100, 1, 1, 1]]
dna_default = [[-1, 320, 0, 3, 1, 1, 0], [-1, 320, 1, 3, 3, 1, 1], [-1, 384, 1, 6, 1, 1, 1], [-1, 448, 1, 9, 1, 1, 1], [-1, 512, 1, 12, 1, 1, 1], [-1, 576, 1, 15, 1, 1, 1], [-1, 640, 1, 18, 1, 1, 1], [-1, 704, 1, 21, 1, 1, 1], [-1, 768, 1, 24, 1, 1, 1], [-1, 832, 1, 27, 1, 1, 1], [-1, 896, 1, 30, 1, 1, 1], [-1, 960, 1, 33, 1, 1, 1]]

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        self.orig = d

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

class dataloader(Dataset):
    def __init__(self, recfile="/tmp/train.hdf5", seq_len=4096, index=False, elen=342):
        self.recfile = recfile
        self.seq_len = seq_len
        self.index = index
        h5 = h5py.File(self.recfile, "r")
        self.len = len(h5["events"])
        h5.close()
        self.elen = elen
        print("Dataloader total events:", self.len, "seqlen:", self.seq_len, "event len:", self.elen)

    def __getitem__(self, index):
        h5 = h5py.File(self.recfile, "r")
        event = h5["events"][index]
        event_len = self.elen
        label = h5["labels"][index]
        label_len = h5["labels_len"][index]
        h5.close()
        if not self.index:
            return (event, event_len, label, label_len)
        else:
            return (event, event_len, label, label_len, index)

    def __len__(self):
        return self.len

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x *( torch.tanh(torch.nn.functional.softplus(x)))

class squeeze_excite(torch.nn.Module):
    def __init__(self, in_channels = 512, size=1, reduction="/16", activation=torch.nn.GELU):
        super(squeeze_excite, self).__init__()
        self.in_channels = in_channels
        self.avg = torch.nn.AdaptiveAvgPool1d(1)
        if type(reduction) == str:
            self.reductionsize = self.in_channels // int(reduction[1:])
        else:
            self.reductionsize = reduction
        self.fc1 = nn.Linear(self.in_channels, self.reductionsize)
        self.activation = activation() # was nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.reductionsize, self.in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg(x)
        x = x.permute(0,2,1)
        x = self.activation(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return input * x.permute(0,2,1)


class convblock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, seperable=True, expansion=True, batchnorm=True, dropout=0.1, activation=torch.nn.GELU, sqex=True, squeeze=32, sqex_activation=torch.nn.GELU, residual=True):
        # no bias?
        super(convblock, self).__init__()
        self.seperable = seperable
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.activation = activation
        self.squeeze = squeeze
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.doexpansion = expansion
        # fix self.squeeze
        dwchannels = in_channels
        if seperable:
            if self.doexpansion and self.in_channels != self.out_channels:
                self.expansion = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=False)
                self.expansion_norm = torch.nn.BatchNorm1d(out_channels)
                self.expansion_act = self.activation()
                dwchannels = out_channels 

            self.depthwise = torch.nn.Conv1d(dwchannels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=out_channels//groups)
            if self.batchnorm:
                self.bn1 = torch.nn.BatchNorm1d(out_channels)
            self.act1 = self.activation()
            if self.squeeze:
                self.sqex = squeeze_excite(in_channels=out_channels, reduction=self.squeeze, activation=sqex_activation)
            self.pointwise = torch.nn.Conv1d(out_channels, out_channels, kernel_size=1, dilation=dilation, bias=bias, padding=0)
            if self.batchnorm:
                self.bn2 = torch.nn.BatchNorm1d(out_channels)
            self.act2 = self.activation()
            if self.dropout:
                self.drop = torch.nn.Dropout(self.dropout)
        else:
            self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
            if self.batchnorm:
                self.bn1 = torch.nn.BatchNorm1d(out_channels)
            self.act1 = self.activation()
            if self.squeeze:
                self.sqex = squeeze_excite(in_channels=out_channels, reduction=self.squeeze, activation=sqex_activation)
            if self.dropout:
                self.drop = torch.nn.Dropout(self.dropout)
        if self.residual and self.stride == 1:
            self.rezero = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, x):
        orig = x

        if self.seperable:
            if self.in_channels != self.out_channels and self.doexpansion:
                x = self.expansion(x)
                x = self.expansion_norm(x)
                x = self.expansion_act(x)
            x = self.depthwise(x)
            if self.batchnorm: x = self.bn1(x)
            x = self.act1(x)
            if self.squeeze:
                x = self.sqex(x)
            x = self.pointwise(x)
            if self.batchnorm: x = self.bn2(x)
            x = self.act2(x) 
            if self.dropout: x = self.drop(x)
        else:
            x = self.conv(x)
            if self.batchnorm: x = self.bn1(x)
            x = self.act1(x)
            if self.dropout: x = self.drop(x)

        if self.residual and self.stride == 1 and self.in_channels == self.out_channels and x.shape[2] == orig.shape[2]:
            return orig + self.rezero * x # rezero
            #return orig + x # normal residual
        else:
            return x

def activation_function(activation):
    if activation == "mish":
        return Mish
    elif activation == "swish":
        return Swish
    elif activation == "relu":
        return torch.nn.ReLU
    elif activation == "gelu":
        return torch.nn.GELU
    else:
        print("Unknown activation type:", activation)
        sys.exit(1)
    
class network(nn.Module):
    def __init__(self, config=None, arch=None, seqlen=4096, debug=False):
        super().__init__()
        if debug: print("Initializing network")
        
        self.seqlen = seqlen
        self.vocab = config.vocab
        
        self.bn = nn.BatchNorm1d

        # [P, Channels, Separable, kernel_size, stride, sqex, dropout]
        # P = -1 kernel_size//2, 0 none, >0 used as padding
        # Channels
        # seperable = 0 False, 1 True
        # kernel_size
        # stride
        # sqex = 0 False, 1 True
        # dropout = 0 False, 1 True
        if arch == None: arch = rna_default

        activation = activation_function(config.activation.lower())
        sqex_activation = activation_function(config.sqex_activation.lower())

        self.convlayers = nn.Sequential()
        in_channels = 1
        convsize = self.seqlen

        for i, layer in enumerate(arch):
            paddingarg = layer[0]
            out_channels = layer[1]
            seperable = layer[2] 
            kernel = layer[3]
            stride = layer[4]
            sqex = layer[5]
            dodropout = layer[6]
            expansion = True

            if dodropout: dropout = config.dropout
            else: dropout = 0
            if sqex: squeeze = config.sqex_reduction
            else: squeeze = 0

            if paddingarg == -1:
                padding = kernel // 2
            else: padding = paddingarg
            if i == 0: expansion = False

            convsize = (convsize + (padding*2) - (kernel-stride))//stride
            if debug:
                print("padding:", padding, "seperable:", seperable, "ch", out_channels, "k:", kernel, "s:", stride, "sqex:", sqex, "drop:", dropout, "expansion:", expansion)
                print("convsize:", convsize)
            self.convlayers.add_module("conv"+str(i), convblock(in_channels, out_channels, kernel, stride=stride, padding=padding, seperable=seperable, activation=activation, expansion=expansion, dropout=dropout, squeeze=squeeze, sqex_activation=sqex_activation, residual=True))
            in_channels = out_channels
            self.final_size = out_channels
         
        self.final = nn.Linear(self.final_size, len(self.vocab))
        if debug: print("Finished init network")

    def forward(self, x):
        #x = self.embedding(x)
        x = self.convlayers(x)
        x = x.permute(0,2,1)
        x = self.final(x)
        x = torch.nn.functional.log_softmax(x, 2)
        return x.permute(1, 0, 2)

counter = 0

def tensorboard_writer_values(writer, model):
    global counter
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.data, counter)
    counter+=1

def tensorboard_writer_value(writer, name, value):
    writer.add_scalar(name, value)

def get_checkpoint(epoch, model, optimizer, scheduler):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
        }
    return checkpoint

def get_config(model, config):
    config = {
            "state_dict": model.state_dict(),
            "config": config
            }
    return config

def train(config = None, args = None, arch = None):
    graph = False
    modelfile = args.model
    trainloss = []
    validloss = []
    learningrate = []
    
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True
    #torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Using training file:", config.trainfile)

    model = network(config=config, arch=arch, seqlen=config.seqlen).to(device)

    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    if modelfile != None:
        print("Loading pretrained model:", modelfile)
        model.load_state_dict(torch.load(modelfile))

    if args.verbose:
        print("Optimizer:", config.optimizer, "lr:", config.lr, "weightdecay", config.weightdecay)
        print("Scheduler:", config.scheduler, "patience:", config.scheduler_patience, "factor:", config.scheduler_factor, "threshold", config.scheduler_threshold, "minlr:", config.scheduler_minlr, "reduce:", config.scheduler_reduce)

    if config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr, weight_decay=config.weightdecay)
    elif config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    elif config.optimizer.lower() == "ranger":
        from pytorch_ranger import Ranger
        optimizer = Ranger(model.parameters(), lr=config.lr, weight_decay=config.weightdecay)

    if args.verbose: print(model)

    model.eval()
    with torch.no_grad():
        fakedata = torch.rand((1, 1, config.seqlen))
        fakeout = model.forward(fakedata.to(device))
        elen = fakeout.shape[0]

    data = dataloader(recfile=config.trainfile, seq_len=config.seqlen, elen=elen)
    data_loader = DataLoader(dataset=data, batch_size=config.batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)

    if config.scheduler == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.scheduler_patience, factor=config.scheduler_factor, verbose=args.verbose, threshold=config.scheduler_threshold, min_lr=config.scheduler_minlr)
    
    count = 0
    last = None

    if config.amp:
        print("Using amp")
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    if args.statedict:
        print("Loading pretrained model:", args.statedict)
        checkpoint = torch.load(args.statedict)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    # from Bonito but weighting for blank changed to 0.1 from 0.4
    if args.labelsmoothing:
        C = len(config.vocab)
        smoothweights = torch.cat([torch.tensor([0.1]), (0.1 / (C - 1)) * torch.ones(C - 1)]).to(device)

    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)

    shutil.rmtree(args.savedir+"/"+config.name, True)
    if args.tensorboard:
        writer = SummaryWriter(args.savedir+"/"+config.name)
        if not graph:
            a,b,c,d = next(iter(data_loader))
            a = torch.unsqueeze(a, 1)
            writer.add_graph(model, a.to(device))

    #criterion = nn.CTCLoss(reduction="mean", zero_infinity=True) # test

    for epoch in range(config.epochs):
        model.train()
        totalloss = 0
        loopcount = 0
        learningrate.append(optimizer.param_groups[0]['lr'])
        if args.verbose: print("Learning rate:", learningrate[-1])

        for i, (event, event_len, label, label_len) in enumerate(data_loader):
            event = torch.unsqueeze(event, 1)
            if event.shape[0] < config.batchsize: continue

            label = label[:, :max(label_len)]
            event = event.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            event_len = event_len.to(device, non_blocking=True)
            label_len = label_len.to(device, non_blocking=True)

            optimizer.zero_grad()

            out = model.forward(event)

            if args.labelsmoothing:
                losses = ont.ctc_label_smoothing_loss(out, label, label_len, smoothweights)
                loss = losses["ctc_loss"]
            else:
                loss = torch.nn.functional.ctc_loss(out, label, event_len, label_len, reduction="mean", blank=config.vocab.index('<PAD>'), zero_infinity=True)
                #loss = criterion(out, label, event_len, label_len)

            totalloss+=loss.cpu().detach().numpy()
            print("Loss", loss.data, "epoch:", epoch, count, optimizer.param_groups[0]['lr'])

            if config.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                if args.labelsmoothing:
                    losses["loss"].backward()
                else:
                    loss.backward()

            if config.gradclip:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradclip)

            optimizer.step()
            loopcount+=1
            count+=1
            if loopcount >= config.train_loopcount: break

        if args.tensorboard: tensorboard_writer_values(writer, model)
        if args.verbose: print("Train epoch loss", totalloss/loopcount)

        vl = validate(model, device, config=config, args=args, epoch=epoch, elen=elen)

        if config.scheduler == "reducelronplateau":
            scheduler.step(vl)
        elif config.scheduler == "decay":
            if (epoch > 0) and (epoch % config.scheduler_reduce == 0):
                optimizer.param_groups[0]['lr'] *= config.scheduler_factor
                if optimizer.param_groups[0]['lr'] < config.scheduler_minlr:
                    optimizer.param_groups[0]['lr'] = config.scheduler_minlr

        trainloss.append(np.float(totalloss/loopcount))
        validloss.append(vl)

        if args.tensorboard:
            tensorboard_writer_value(writer, "training loss", np.float(totalloss/loopcount))
            tensorboard_writer_value(writer, "validation loss", vl)

        f = open(args.savedir+"/"+config.name+"-stats.pickle", "wb")
        pickle.dump([trainloss, validloss], f)
        pickle.dump(config.orig, f)
        pickle.dump(learningrate, f)
        f.close()

        torch.save(get_config(model, config.orig), args.savedir+"/"+config.name+"-epoch"+str(epoch)+".torch")
        torch.save(get_checkpoint(epoch, model, optimizer, scheduler), args.savedir+"/"+config.name+"-ext.torch")

        if args.verbose:
            print("Train losses:", trainloss)
            print("Valid losses:", validloss) 
            print("Learning rate:", learningrate)

    print("Model", config.name, "done.")
    return trainloss, validloss

def validate(model, device, config = None, args=None, epoch=-1, elen=342):
    if config.valid_loopcount < 1: return(np.float(0))
    modelfile = None
    if args != None: modelfile = args.model
    print("Running validation")

    # NOTE: possibly move these into train
    valid_data = dataloader(recfile=config.validfile, seq_len=config.seqlen, elen=elen)
    valid_loader = DataLoader(dataset=valid_data, batch_size=config.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    total = 0
    totalloss = 0

    if model is None and modelfile:
        model = network(config=config).to(device)
        model.load_state_dict(torch.load(modelfile))

    model.eval()

    with torch.no_grad():
        for i, values in enumerate(valid_loader):
            event = values[0]
            event_len = values[1]
            label = values[2]
            label_len = values[3]
            event = torch.unsqueeze(event, 1)
            if event.shape[0] < config.batchsize: continue

            label = label[:, :max(label_len)]
            event = event.to(device)
            event_len = event_len.to(device)
            label = label.to(device)
            label_len = label_len.to(device)

            out = model.forward(event)
            loss = torch.nn.functional.ctc_loss(out, label, event_len, label_len, reduction="mean", blank=0, zero_infinity=True)
            totalloss += loss.cpu().detach().numpy()

            print("valid loss", loss)
            total+=1
            if total >= config.valid_loopcount: break

        print("Validation loss:", totalloss / total)

    return np.float(totalloss / total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data.')
    parser.add_argument("-c", "--config", default=None, type=str)
    parser.add_argument("-d", "--statedict", default=None, type=str)
    parser.add_argument("-a", "--arch", default=None, type=str, help="Architecture file")
    parser.add_argument("-D", "--savedir", default="runs", type=str, help="save directory (default: runs)")
    parser.add_argument("-n", "--name", default=None, type=str)
    parser.add_argument("-m", "--model", default=None, type=str)
    parser.add_argument("-l", "--labelsmoothing", default=False, action="store_true")
    parser.add_argument("-t", "--tensorboard", default=False, action="store_true", help="turn tensorboard logging on")
    parser.add_argument("-w", "--workers", default=8, type=int, help="num_workers (default: 8)")
    parser.add_argument("-v", "--verbose", default=True, action="store_false", help="Turn verbose mode off")
    parser.add_argument("--rna", default=False, action="store_true", help="Use default RNA model")
    parser.add_argument("--dna", default=False, action="store_true", help="Use default RNA model")
    args = parser.parse_args()

    if args.name == None:
        args.name = ""
        while args.name == "": args.name = input("Number: ")
    defaultconfig["name"] = args.name 

    if args.config != None:
        import yaml, re

        # https://stackoverflow.com/questions/52412297/how-to-replace-environment-variable-value-in-yaml-file-to-be-parsed-using-python
        def path_constructor(loader, node): 
            #print(node.value) 
            return os.path.expandvars(node.value) 

        class EnvVarLoader(yaml.SafeLoader):
            pass

        path_matcher = re.compile(r'.*\$\{([^}^{]+)\}.*')
        EnvVarLoader.add_implicit_resolver('!path', path_matcher, None)
        EnvVarLoader.add_constructor('!path', path_constructor)

        newconfig = yaml.load(open(args.config), Loader=EnvVarLoader)
        defaultconfig.update(newconfig)

    if args.arch != None:
        defaultconfig["arch"] = open(args.arch, "r").read()

    if args.arch != None:
        print("Loading architecture from:", args.arch)
        args.arch = eval(open(args.arch, "r").read())

    if args.rna:
        args.arch = rna_default

    if args.dna:
        args.arch = dna_default

    config = objectview(defaultconfig)
    train(config=config, args=args, arch=args.arch)
