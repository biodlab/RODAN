#!/usr/bin/env python
#
# RODAN
# v1.0
# (c) 2020,2021,2022 Don Neumann
#
# for both RNA and DNA:
# minimap2 --secondary=no -ax map-ont -t 32 --cs genomefile fastafile > file.sam
#

import re, sys, argparse, pysam
import numpy as np


def parse_cs_tag(tag, refseq, readseq, debug=False):
    ret = ""
    match = 0
    mismatch = 0
    deletions = 0
    insertions = 0
    refpos = 0
    readpos = 0
    refstr = ""
    readstr = ""
    p = re.findall(r":([0-9]+)|(\*[a-z][a-z])|(=[A-Za-z]+)|(\+[A-Za-z]+)|(\-[A-Za-z]+)", tag)
    for i, x in enumerate(p):
        #print(x)
        if len(x[0]):
            q = int(x[0])
            if debug: print("match:", i, q)
            match+= int(q)
            refstr+=refseq[refpos:refpos+q]
            refpos+=q
            readstr+=readseq[readpos:readpos+q]
            readpos+=q
            ret+= "|" * q
        if len(x[1]):
            q = int((len(x[1])-1)/2)
            if debug: print("mismatch:", i, q)
            mismatch+= q
            refstr+=refseq[refpos:refpos+q]
            refpos+=q
            readstr+=readseq[readpos:readpos+q]
            readpos+=q
            ret+= "*" * q
        if len(x[2]):
            if debug: print("FAIL")
        if len(x[3]):
            q = len(x[3])-1
            if debug: print("insertion:", i, q, x[3])
            insertions+=q
            refstr+="-"*q
            readstr+=readseq[readpos:readpos+q]
            readpos+=q
            ret+= " " * q
        if len(x[4]):
            q = len(x[4])-1
            if debug: print("deletion:", i, q)
            deletions+=q
            refstr+=refseq[refpos:refpos+q]
            refpos+=q
            readstr+="-"*q
            ret+= " " * q
    if debug:
        print("match:", match, "mismatch:", mismatch, "deletions:", deletions, "insertions:", insertions)
        print(len(refstr), len(ret), len(readstr))
        print(refstr[:80])
        print(ret[:80])
        print(readstr[:80])
    tot = match+mismatch+deletions+insertions
    totfail = mismatch+deletions+insertions
    return tot, totfail, match/tot, mismatch/tot, deletions/tot, insertions/tot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="samparse")
    parser.add_argument("samfile", default=None, type=str)
    parser.add_argument("genomefile", default=None, type=str)
    parser.add_argument("-k", "--keepall", default=False, action="store_true", help="do not discard supplementary alignments")
    args = parser.parse_args()

    fastafile = pysam.FastaFile(args.genomefile)

    arr = []

    sf = pysam.AlignmentFile(args.samfile, "r")
    for read in sf.fetch():
        if read.is_unmapped: continue
        if not args.keepall and (read.flag & 256 or read.flag & 2048): continue
        refseq = fastafile.fetch(read.reference_name, read.reference_start, read.reference_end)
        if read.seq == None:
            print("FAIL:", read.qname, read.reference_name)
            continue
        seq = read.seq[read.query_alignment_start:read.query_alignment_end]
        cstag = read.get_tag("cs")
        ans = parse_cs_tag(cstag, refseq, seq)
        #print(read.qname+","+read.reference_name+","+",".join(map(str, ans[2:])))
        arr.append([ans[2], ans[3], ans[4], ans[5]])
    arr = np.array(arr)
    print("Total:", len(arr), "Median accuracy:", np.median(arr[:,0]), "Average accuracy:", np.mean(arr[:,0]), "std:", np.std(arr[:,0]))
    print("Median  - Mismatch:", np.median(arr[:,1]), "Deletions:", np.median(arr[:,2]), "Insertions:", np.median(arr[:,3]))
    print("Average - Mismatch:", np.mean(arr[:,1]), "Deletions:", np.mean(arr[:,2]), "Insertions:", np.mean(arr[:,3]))

