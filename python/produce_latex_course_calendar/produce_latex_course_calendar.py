import datetime as dt
import time

year = 2014
starting_month = 1
starting_day = 21

datestring = "%d %d %d" % (year, starting_month, starting_day)
start = time.strptime(datestring,"%Y %m %d")

year_day = start.tm_yday

tuesday = True

i=0
week = 0
while i < 110:

    day = year_day + i
    datestring = "%d %d" % (year, day)
    start = time.strptime(datestring,"%Y %j")

    #print time.strftime("%Y %b %d",start)
    date = time.strftime("%a., %b %d",start)
    if tuesday:
        print "Week %0d & %s & XXX & YYY & ZZZ \\\\ " % (week,date)
        #print "\\hline"
    elif thursday:
        print "         & %s & XXX & YYY & ZZZ \\\\ " % (date)
        print "\\hline"
    else:
        print "        & %s  & XXX & YYY & ZZZ \\\\" % (date)
        print "\\hline"

    if tuesday:
        increment = 2
        #increment = 1
        #increment = 7
        tuesday = False
        #wednesday = False
        thursday = True
        #week += 1
    #elif wednesday:
        #increment = 1
        #tuesday = False
        #wednesday = False
        #thursday = True
    elif thursday:
        increment = 5
        tuesday = True
        #wednesday = False
        thursday = False
        week += 1
    #else:
        #increment = 5
        #tuesday = True
        #wednesday = False
        #thursday = False
        #week += 1

    i += increment



