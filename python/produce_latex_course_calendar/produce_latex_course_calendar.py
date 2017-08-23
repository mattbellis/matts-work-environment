import datetime as dt
import time

year = 2017
starting_month = 9
starting_day = 6

datestring = "%d %d %d" % (year, starting_month, starting_day)
start = time.strptime(datestring,"%Y %m %d")

year_day = start.tm_yday

monday = False
tuesday = False
wednesday = True
thursday = False
friday = False

i=0
week = 0
while i < 120:

    day = year_day + i
    datestring = "%d %d" % (year, day)
    start = time.strptime(datestring,"%Y %j")

    #print time.strftime("%Y %b %d",start)
    date = time.strftime("%a., %b %d",start)
    if monday:
        #print("Week %0d  & %s & XXX & YYY & ZZZ \\\\ " % (week,date))
        print("Week %0d  & %s & XXX \\\\ " % (week,date))
        #print "\\hline"
    elif wednesday:
        #print("         & %s & XXX & YYY & ZZZ \\\\ " % (date))
        print("         & %s & XXX \\\\ " % (date))
    else:
        #print("         & %s & XXX & YYY & ZZZ \\\\ " % (date))
        print("         & %s & XXX \\\\ " % (date))
        print("\\hline")
    '''
    else:
        print "        & %s  & XXX & YYY & ZZZ \\\\" % (date)
        print "\\hline"
    '''

    if monday:
        increment = 2
        monday = False
        tuesday = False
        wednesday = True
        thursday = False
        friday = False
        #week += 1
    elif tuesday:
        increment = 2
        monday = False
        tuesday = False
        wednesday = True
        thursday = False
        friday = False
        #week += 1
    elif wednesday:
        increment = 2
        monday = False
        tuesday = False
        wednesday = False
        thursday = False
        friday = True
    elif thursday:
        increment = 5
        monday = False
        tuesday = False
        wednesday = True
        thursday = False
        friday = False
        #week += 1
    elif friday:
        increment = 3
        monday = True
        tuesday = False
        wednesday = False
        thursday = False
        friday = False
        week += 1
    #else:
        #increment = 5
        #tuesday = True
        #wednesday = False
        #thursday = False
        #week += 1

    i += increment



