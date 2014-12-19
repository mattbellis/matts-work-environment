import sys
import csv 
from pylab import *
import matplotlib.pyplot as plt
import argparse

import datetime

import plotly
import plotly.plotly as py
from plotly.graph_objs import Box,Layout

################################################################################
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



################################################################################
# main
################################################################################
def main():

    ############################################################################
    # Parse the arguments
    ############################################################################
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_file_name', type=str, default=None, 
            help='Input file name')
    '''
    parser.add_argument('--dump-names', dest='dump_names', action='store_true',
            default=True,
            help='Dump the names of the students and an index for each.')
    parser.add_argument('--dump-grades', dest='dump_grades', action='store_true',
            default=False,
            help='Dump the grades for all the students.')
    parser.add_argument('--gvis', dest='gvis', action='store_true',
            default=False,
            help='Dump output for the Google Motion Chart.')
    parser.add_argument('--password', dest='password', default=None, 
            help='Password for mail server.')
    parser.add_argument('--course', dest='course', default=None, 
            help='Course name.')
    parser.add_argument('--student', dest='student', default=None, type=int,
            help='Student name to dump info for or email.')
    '''

    args = parser.parse_args()
    #print args

    ############################################################################

    if args.input_file_name is None:
        print "Must pass in an input file name!"
        parser.print_help()

    filename = args.input_file_name
    infile = csv.reader(open(filename, 'rb'), delimiter=',', quotechar='#')

    ############################################################################
    #py.sign_in("MatthewBellis", "d6h4et78v5")
    ############################################################################

    print infile
    courses = []
    icourses = []
    topics = {}
    ncourses = 0
    for line_num,row in enumerate(infile):
        #print row
        if line_num==0:
            for i,r in enumerate(row):
                r = r.strip()
                print r
                if r is not '':
                    courses.append(r)
                    icourses.append(i)
            ncourses = len(courses)
        else:
            topic = "%s %s" % (row[0],row[1])
            print topic
            if topic is not ' ':
                values = np.zeros(ncourses)
                for j,index in enumerate(icourses):
                    if is_number(row[index]):
                        values[j] = float(row[index])
                topics[topic] = values
                
                

    print courses
    print topics

    fig = plt.figure()
    fig.add_subplot(1,1,1)
    yvals = np.arange(1,len(courses)+1,1)
    icount = 0
    for key, value in topics.iteritems():
        # plt.plot(key,value)
        xvals = np.ones(len(courses)) + icount
        print xvals
        print yvals
        print value
        value = np.array(value)
        print len(xvals)
        print len(yvals)
        print len(value)
        max_val = max(value)
        value[value!=max_val] = 0
        plt.scatter(xvals,yvals,s=10*(value**3),alpha=0.5)
        #plt.plot(xvals,yvals,'o')
        #print key
        #print value
        icount += 1

    plt.yticks(yvals, courses)
    xvals = np.arange(1,len(topics)+1,1)
    plt.xticks(xvals, topics.keys(),rotation='vertical')
    plt.subplots_adjust(bottom=0.30)
    plt.show()

################################################################################
if __name__ == "__main__":
    main()

