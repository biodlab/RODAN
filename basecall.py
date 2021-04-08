#!/usr/bin/env python
#
# RODAN
# (c) 2020,2021 Don Neumann
#

import torch
import numpy as np
import os, sys, re, argparse, pickle, time, glob
from torch.utils.data import Dataset, DataLoader
from fast_ctc_decode import beam_search
from ont_fast5_api.fast5_interface import get_fast5_file
from torch.multiprocessing import Queue, Process

import model as network
import ont


def segment(seg, s):
    seg = np.concatenate((seg, np.zeros((-len(seg)%s))))
    nrows=((seg.size-s)//s)+1
    n=seg.strides[0]
    return np.lib.stride_tricks.as_strided(seg, shape=(nrows,s), strides=(s*n, n))

def convert_statedict(state_dict):
    from collections import OrderedDict
    new_checkpoint = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_checkpoint[name] = v
    return new_checkpoint

def load_model(modelfile, config = None, args = None):
    if config.amp:
        from apex import amp
        if args.debug: print("Using apex amp")
    if modelfile == None:
        sys.stderr.write("No model file specified!")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.debug: print("Using device:", device)
    if args.arch is not None:
        model = network.network(config=config, arch=args.arch).to(device)
    else:
        model = network.network(config=config).to(device)
    if args.debug: print("Loading pretrained weights:", modelfile)
    state_dict = torch.load(modelfile)["state_dict"]
    if "state_dict" in state_dict:
        model.load_state_dict(convert_statedict(state_dict["state_dict"]))
    else:
        model.load_state_dict(torch.load(modelfile)["state_dict"])
    if args.debug: print(model)

    model.eval()
    torch.set_grad_enabled(False)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    if config.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    return model, device

def fast5_data(file):
    f = get_fast5_file(file, "r")
    reads = f.get_read_ids()
    if len(reads) > 1: 
        sys.stderr.write("ERROR: Designed only for single read fast5 files")
        sys.exit(1)
    read = f.get_read(reads[0])
    signal = f.get_raw_data()
    r = read.handle[read.global_key + "channel_id"].attrs["range"]
    d = read.handle[read.global_key + "channel_id"].attrs["digitisation"]
    o = read.handle[read.global_key + "channel_id"].attrs["offset"]
    signal = (signal + o) * r / d
    f.close()
    return signal

def get_fast5(dir):
    for file in glob.iglob(dir+"/**/*.fast5", recursive=True):
        yield file

def mp_files(dir, queue, config, args):
    chunkname = []
    chunks = None
    queuechunks = None
    chunkremainder = None
    for file in glob.iglob(dir+"/**/*.fast5", recursive=True):
        while queue.qsize() >= 100:
            time.sleep(1)
        outfile = os.path.splitext(os.path.basename(file))[0]
        try:
            signal = np.array(fast5_data(file)).astype(np.float32)
            if args.debug: print("mp_files:", file)
        except:
            continue
        signal_start = 0
        signal_end = len(signal)
        med, mad = ont.med_mad(signal[signal_start:signal_end])
        signal = (signal[signal_start:signal_end] - med) / mad
        newchunks = segment(signal, config.seqlen)
        if chunks is not None:
            chunks = np.concatenate((chunks, newchunks), axis=0)
            queuechunks += [file] * newchunks.shape[0]
        else:
            chunks = newchunks
            queuechunks = [file] * newchunks.shape[0]
        if chunks.shape[0] >= args.batchsize:
            for i in range(0, chunks.shape[0]//args.batchsize, args.batchsize):
                queue.put((queuechunks[:args.batchsize], chunks[:args.batchsize]))
                chunks = chunks[args.batchsize:]
                queuechunks = queuechunks[args.batchsize:]
    if len(queuechunks) > 0:
        if args.debug: print("queuechunks:", len(queuechunks), chunks.shape[0])
        for i in range(0, int(np.ceil(chunks.shape[0]/args.batchsize)), args.batchsize):
            start = i * args.batchsize
            end = start + args.batchsize
            if end > chunks.shape[0]: end = chunks.shape[0]
            queue.put((queuechunks[start:end], chunks[start:end]))
            if args.debug: print("put last chunk", chunks[start:end].shape[0])

    queue.put(("end", None))


def mp_gpu(inqueue, outqueue, config, args):
    model, device = load_model(args.model, config, args)
    shtensor = None
    while True:
        time1 = time.perf_counter()
        read = inqueue.get()
        file = read[0]
        if type(file) == str: 
            outqueue.put(("end", None))
            break
        chunks = read[1]
        for i in range(0, chunks.shape[0], config.batchsize):
            end = i+config.batchsize
            if end > chunks.shape[0]: end = chunks.shape[0]
            event = torch.unsqueeze(torch.FloatTensor(chunks[i:end]), 1).to(device, non_blocking=True)
            out=model.forward(event)
            if shtensor is None:
                shtensor = torch.empty((out.shape), pin_memory=True, dtype=out.dtype)
            if out.shape[1] != shtensor.shape[1]:
                shtensor = torch.empty((out.shape), pin_memory=True, dtype=out.dtype)
            logitspre = shtensor.copy_(out).numpy()
            if args.debug: print("mp_gpu:", logitspre.shape)
            outqueue.put((file, logitspre))
            del out
            del logitspre

def mp_write(queue, config, args):
    files = None
    chunks = None
    totprocessed = 0
    finish = False
    while True:
        if queue.qsize() > 0:
            newchunk = queue.get()
            if type(newchunk[0]) == str:
                if not len(files): break
                finish = True
            else:
                if chunks is not None:
                    chunks = np.concatenate((chunks, newchunk[1]), axis=1)
                    files = files + newchunk[0]
                else:
                    chunks = newchunk[1]
                    files = newchunk[0]
            while files.count(files[0]) < len(files) or finish:
                totlen = files.count(files[0])
                callchunk = chunks[:,:totlen,:]
                logits = np.transpose(np.argmax(callchunk, -1), (1, 0))
                label_blank = np.zeros((logits.shape[0], logits.shape[1] + 200))
                try:
                    out,outstr = ctcdecoder(logits, label_blank, pre=callchunk, beam_size=args.beamsize)
                except:
                    # failure in decoding
                    out = ""
                    outstr = ""
                    pass
                seq = ""
                if len(out) != len(outstr):
                    sys.stderr.write("FAIL:", len(out), len(outstr), files[0])
                    sys.exit(1)
                for j in range(len(out)):
                    seq += outstr[j]
                readid = os.path.splitext(os.path.basename(files[0]))[0]
                print(">"+readid)
                if args.reverse:
                    print(seq[::-1])
                else:
                    print(seq)
                newchunks = chunks[:,totlen:,:]
                chunks = newchunks
                files = files[totlen:]
                totprocessed+=1
                if finish and not len(files): break
            if finish: break
                
vocab = { 1:"A", 2:"C", 3:"G", 4:"T" }

def ctcdecoder(logits, label, blank=False, beam_size=5, alphabet="NACGT", pre=None):
    ret = np.zeros((label.shape[0], label.shape[1]+50))
    retstr = []
    for i in range(logits.shape[0]):
        if pre is not None:
            beamcur = beam_search(torch.softmax(torch.tensor(pre[:,i,:]), dim=-1).cpu().detach().numpy(), alphabet=alphabet, beam_size=beam_size)[0]
        prev = None
        cur = []
        pos = 0
        for j in range(logits.shape[1]):
            if not blank:
                if logits[i,j] != prev:
                    prev = logits[i,j]
                    try:
                        if prev != 0:
                            ret[i, pos] = prev
                            pos+=1
                            cur.append(vocab[prev])
                    except:
                        sys.stderr.write("ctcdecoder: fail on i:", i, "pos:", pos)
            else:
                if logits[i,j] == 0: break
                ret[i, pos] = logits[i,j] # is this right?
                cur.append(vocab[logits[i,pos]])
                pos+=1
        if pre is not None:
            retstr.append(beamcur)
        else:
            retstr.append("".join(cur))
    return ret, retstr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basecall fast5 files')
    parser.add_argument("fast5dir", default=None, type=str)
    parser.add_argument("-a", "--arch", default=None, type=str, help="architecture settings")
    parser.add_argument("-m", "--model", default="rna.torch", type=str, help="default: rna.torch")
    parser.add_argument("-r", "--reverse", default=True, action="store_true", help="reverse for RNA (default: True)")
    parser.add_argument("-b", "--batchsize", default=200, type=int, help="default: 200")
    parser.add_argument("-B", "--beamsize", default=5, type=int, help="CTC beam search size (default: 5)")
    parser.add_argument("-e", "--errors", default=False, action="store_true")
    parser.add_argument("-d", "--debug", default=False, action="store_true")
    args = parser.parse_args()

    torchdict = torch.load(args.model, map_location="cpu")
    origconfig = torchdict["config"]

    if args.debug: print(origconfig)
    origconfig["debug"] = args.debug
    config = network.objectview(origconfig)
    config.batchsize = args.batchsize

    if args.arch != None:
        if args.debug: print("Loading architecture from:", args.arch)
        args.arch = eval(open(args.arch, "r").read())
    else:
        args.arch = eval(config.arch)

    if args.debug: print("Using sequence len:", int(config.seqlen))
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    call_queue = Queue()
    write_queue = Queue()
    p1 = Process(target=mp_files, args=(args.fast5dir, call_queue, config, args,))
    p2 = Process(target=mp_gpu, args=(call_queue, write_queue, config, args,))
    p3 = Process(target=mp_write, args=(write_queue, config, args,))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
