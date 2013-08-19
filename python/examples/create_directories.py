#!/usr/bin/env python

# Make some directories

import sys
import os
import subprocess as sp


################################################################################
# Make some subdirectories
################################################################################

directory_name = "test_dir"
link_name = "test_dir_link"

subdir_names = ['bin','lib','src']

# Check to see if directory exists.
# If it does not, then make it.
if not os.access( directory_name, os.W_OK ):
    os.mkdir(directory_name,0744)
else:
    if not os.access( link_name, os.W_OK ):
        os.symlink(directory_name, link_name)
        print "Directory already exists, making a soft link instead."


#############################################
# Make the subdirs
# Number them. Make 10 different versions.
#############################################
for i in range(0,10):
    for dir in subdir_names:

        name = "%s_%04d" % (dir,i)

        if not os.access( name, os.W_OK ):
            os.mkdir(name,0744)
        else:
            print "Directory already exists!  %s" % (name)

################################################################################
# List what's in the subdirectories.
# This is an example of how to execute shell commands from inside python.
################################################################################

cmd = ['ls']
sp.Popen(cmd,0).wait()

cmd = ['ls', '-ltr']
sp.Popen(cmd,0).wait()

cmd = ['ls']
output = sp.Popen(cmd, stdout=sp.PIPE).communicate()[0]

names = output.split()
for o in names:
    print "Here's a file or directory: %s" % (o)

