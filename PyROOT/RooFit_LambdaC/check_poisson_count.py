#!/usr/bin/env python

import sys

import ROOT
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import *


def file_len(fname):
  #with open(fname) as f:
  f = open(fname, "r")
  for i, l in enumerate(f):
    pass
  return i + 1

def main():

  nums = []
  outfile = open("num_count.txt", "w+")

  #for n,j in enumerate([0, 20]):
  for n,j in enumerate([0, 20, 50, 60, 100, 200, 500]):
    print j
    nums.append([])
    for i in range(0,1000):
      if j==0:
        infilename = "data/toymc_bkg_wNNvar_test_new_binning_n1500_%04d.dat" % ( i )
      else:
        infilename = "data/toymc_bkg_wNNvar_test_new_binning_n1500_embed_poisson%d_%04d.dat" % ( j, i)
      flen = file_len(infilename)
      #print flen
      nums[n].append(flen)


  for i in range(0,1000):
    output = ""
    for n in xrange(len(nums)):
      output += "%d " % (nums[n][i])
    output += "\n"
    outfile.write(output)

  outfile.close()


if __name__ == "__main__":
  main()
  rep = ''
  while not rep in [ 'q', 'Q' ]:
    rep = raw_input( 'enter "q" to quit: ' )
    if 1 < len(rep):
      rep = rep[0]

