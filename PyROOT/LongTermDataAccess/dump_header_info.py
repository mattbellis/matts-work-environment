#!/usr/bin/env python

import sys
from optparse import OptionParser

# Parse command line options
parser = OptionParser()
parser.add_option("-a", "--all", dest="dump_all", action="store_true", default = False, help="Dump all the information")
parser.add_option("-t", "--tree-name", dest="tree_name", default="T", help="Name of the TTree object")
parser.add_option("-i", "--info", dest="info", action="append", help="Append these choices for specific information to dump out.\nNote that only the # of the information is necessary. e.g: -i 0 -i2")

(options, args) = parser.parse_args()

from ROOT import *

# Open the file and grab the tree
f = TFile(sys.argv[1])
t = gROOT.FindObject(str(options.tree_name))

# Get the user info from the TTree object
user_info = t.GetUserInfo()

###################################################
# Print it out, according to the options
###################################################
header_entries = []
#user_info.Print()
for i,user_lists in enumerate(user_info):
  # Only dump out the first two rows which describe the file itself and
  # what information is stored for each variable.
  if not options.dump_all and i<2:
    output = str(user_lists.GetName()) + "\n"
    for j,user_defs in enumerate(user_lists):
      if i==0:
        output += "\t%s\n" % (str(user_defs))
      else:
        output += "\t%d: %s\n" % (j, str(user_defs))

    print output

  # Dump everything
  elif options.dump_all:
    output = str(user_lists.GetName()) + "\n"
    for j,user_defs in enumerate(user_lists):
      defs = str(user_defs)
      if i<=1:
        if i==1:
          header_entries.append(defs)
        output += "\t%s\n" % (defs)
      else:
        # Dump only certain fields, if specified on the command lines.
        if options.info!=None:
          if str(j) in options.info:
            output += "%20s: %s\n" % (header_entries[j],  defs)
        else:
          output += "%20s: %s\n" % (header_entries[j],  defs)

    print output
  
  
