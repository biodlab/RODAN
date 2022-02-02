#!/usr/bin/env python
#
# RODAN
# v1.0
# (c) 2020,2021,2022 Don Neumann
#

import h5py
import numpy as np
import sys, argparse, pickle, copy

class savefile:
    def __init__(self, file = None, seqlen = 4096, position=False, savelength=False):
        if not file:
            print("savefile error: no file specified")
            sys.exit(1)

        self.seqlen = seqlen
        self.position = position
        self.savelength = savelength
        self.h5 = h5py.File(file, "w")
        self.h5_event = self.h5.create_dataset("events", dtype="float32", shape=(0, self.seqlen), maxshape=(None, self.seqlen))
        if self.savelength:
            self.h5_event_len = self.h5.create_dataset("events_len", dtype="int64", shape=(0,), maxshape=(None,))
        self.h5_label = self.h5.create_dataset("labels", dtype="int64", shape=(0,0), maxshape=(None, self.seqlen))
        self.h5_label_len = self.h5.create_dataset("labels_len", dtype="int64", shape=(0,), maxshape=(None,))
        if self.position:
            self.h5_position = self.h5.create_dataset("position", dtype=h5py.special_dtype(vlen=str), shape=(0,), maxshape=(None,))
        self.maxlabsize = 0
        self.size = 0

    def update(self, events, event_len, label, label_len, events_orig = None, events_digit = None, position = None):
        if events.shape[0] == 0: return
        # x+1 below is to index from 1-4, not 0-3, since 0 is blank in ctc
        label = np.array([x + [0] * (max(label_len)-len(x)) for x in label])
        if max(label_len) > self.maxlabsize:
            self.maxlabsize = max(label_len)
            self.h5_label.resize((self.size, self.maxlabsize))
        if label.shape[1] < self.maxlabsize:
            label = np.pad(label, ((0,0), (0,self.maxlabsize-label.shape[1])), mode="constant", constant_values=0)
        self.h5_event.resize((self.size+events.shape[0]), axis=0)
        self.h5_event[self.size:self.size+events.shape[0]] = events
        if self.savelength:
            self.h5_event_len.resize((self.size+events.shape[0]), axis=0)
            self.h5_event_len[self.size:self.size+events.shape[0]] = event_len
        self.h5_label.resize((self.size+events.shape[0]), axis=0)
        self.h5_label[self.size:self.size+events.shape[0]] = label
        self.h5_label_len.resize((self.size+events.shape[0]), axis=0)
        self.h5_label_len[self.size:self.size+events.shape[0]] = label_len
        if self.position:
            self.h5_position.resize((self.size+events.shape[0]), axis=0)
            self.h5_position[self.size:self.size+events.shape[0]] = position
        self.h5.flush()
        self.size += events.shape[0]

    def count(self):
        return self.size
    
    def close(self):
        self.h5.close()

    def __del__(self):
        self.h5.close()
 
def reader(args, offset=1.4826):
    if args.outdir != None:
        out = savefile(args.outdir+"/train.hdf5", seqlen=args.seqlen, position=args.position, savelength=args.savelength)
    count = 0
    which = 0
    stats = dict()
    statsarr = []
    maxcount = args.maxchunks

    f = h5py.File(args.infile, "r")
    keys = list(f["Reads"].keys())
    print("Total reads:", len(keys))
    np.random.shuffle(keys)
    for read in keys:
        pos = f["Reads"][read]
        s_offset = pos.attrs["offset"]
        s_range = pos.attrs["range"]
        s_dig = pos.attrs["digitisation"]
        signal = (pos["Dacs"][:] + s_offset) * s_range / s_dig
        if len(signal) > args.maxsignallen: continue
        if args.debug: print("Read:", read, len(signal))
        shift = pos.attrs["shift_frompA"] # median
        scale = pos.attrs["scale_frompA"] # mad
        med = np.median(signal)
        mad = offset * np.median(abs(signal-med))
        signal = (signal - shift) / scale
        if args.debug: print("Median:", shift, med, "MAD:", scale, mad)
        ref = pos["Reference"][:]
        r2s = pos["Ref_to_signal"][:]
        
        start = np.random.randint(0, args.start)
        if args.debug:
            print("ref:", ref.shape, "r2s:", r2s.shape, "start:", start)
        events = []
        event_len = []
        events_position = []
        label = []
        label_len = []

        curpos = start
        curloop = 0
        while curpos < len(signal):
            i = curpos
            if not args.randomsize:
                chunklen = args.seqlen
            else:
                if np.random.rand() >= args.samplepct:
                    chunklen = args.seqlen
                else:
                    chunklen = np.random.randint(args.randomsize, args.seqlen)
            if i+chunklen >= len(signal): break
            if args.overlap > 0:
                curpos += chunklen - args.overlap
            else:
                curpos += chunklen
            posleft = i 
            posright = i + chunklen
            r1  = np.searchsorted(r2s, (posleft+args.trim, posright-args.trim))
            begin, end = r1
            bases = end-begin
            # below is for a weird RNA problem
            if len(list(ref[begin:end]+1)) != bases: continue 
            if bases == 0:
                spb = 0
            else:
                spb = chunklen / bases
            if args.stats:
                if np.int(spb) in stats:
                    stats[np.int(spb)] += 1
                else:
                    stats[np.int(spb)] = 1
                statsarr.append(spb)

            if args.dryrun: continue

            if spb < args.minbase or spb > args.maxbase:
                if args.debug: print("skipped: spb is", spb, str(chunklen)+"/"+str(bases))
                continue
            if chunklen != args.seqlen:
                chunk = np.zeros((args.seqlen))
                chunk[0:chunklen] = signal[i:i+chunklen]
            else:
                chunk = signal[i:i+chunklen]
            events.append(chunk)
            if args.position:
                events_position.append(read+":"+str(posleft))
            event_len.append(chunklen)
            label.append(list(ref[begin:end]+1))
            label_len.append(bases)
            if curloop >= args.maxsamples: break
            curloop+=1
        
        if args.dryrun: continue

        events = np.array(events)
        if args.debug:
            print("events final:", events.shape)
        event_len = np.array(event_len)
        label_len = np.array(label_len)
        if args.outdir != None:
            out.update(events, event_len, label, label_len, position=events_position)
        count += events.shape[0]
        if count > maxcount:
            if args.outdir != None: out.close()
            if which >= 1: break
            print("Starting validation file")
            which +=1
            if args.outdir != None:
                out = savefile(args.outdir+"/valid.hdf5", seqlen = args.seqlen, savelength=args.savelength)
            count = 0
            maxcount = args.maxvalchunks
    if args.stats:
        print(stats)
        f = open("stats.pickle", "wb")
        pickle.dump(stats, f)
        pickle.dump(statsarr, f)
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data.')
    parser.add_argument("-i", "--infile", default="/s/jawar/o/nobackup/don-backups/ont-data/bonito.hdf5", type=str)
    parser.add_argument("--outdir", default=None, type=str, help="directory for output files")
    parser.add_argument("--seqlen", default=4096, type=int, help="sequence length, default 1024")
    parser.add_argument("--minbase", default=20, type=int, help="minimum samples per base (default 20)")
    parser.add_argument("--maxbase", default=70, type=int, help="maximum samples per base (default 70)")
    parser.add_argument("-o", "--overlap", default=-1, type=int, help="overlap chunks (default: disabled)")
    parser.add_argument("--maxchunks", default=1000000, type=int, help="chunks of training data (default 1000000)")
    parser.add_argument("--maxvalchunks", default=100000, type=int, help="max chunks of validation data (default 100000)")
    parser.add_argument("--maxsignallen", default=10000000000, type=int, help="use signals up to this length (default: 10^10)")
    parser.add_argument("--dryrun", default=False, action="store_true", help="Stats from all data")
    parser.add_argument("--start", default=1024, type=int, help="Randomly start between 0 and this value (default: 1024)")
    parser.add_argument("-p", "--position", default=False, action="store_true", help="store position information on reads")
    parser.add_argument("-s", "--stats", default=False, action="store_true", help="save stats in stats.pickle")
    parser.add_argument("-t", "--trim", default=0, type=int, help="trim bases to start+trim to end-trim")
    parser.add_argument("-l", "--savelength", default=False, action="store_true", help="enable writing sequence length")
    parser.add_argument("-z", "--zerostats", default=False, action="store_true")
    parser.add_argument("-d", "--debug", default=False, action="store_true")
    parser.add_argument("-r", "--randomsize", default=0, type=int, help="lower bound for random sized chunks (Default: disabled)")
    parser.add_argument("-S", "--samplepct", default=1.0, type=float, help="float percentage of random sized samples (Default: 1.0)")
    parser.add_argument("-m", "--maxsamples", default=10000, type=int, help="maximum amount of chunks per sample (default: 10000)")
    args = parser.parse_args()

    reader(args)
