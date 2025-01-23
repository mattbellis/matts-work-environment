# 
#elec= 10  #int(sys.argv[1])
#ev=   100   #float(sys.argv[2])
#BGP=   800   #float(sys.argv[3])
#nRub=  1.6e13   #float(sys.argv[4])
#rbPol= 99.9  #float(sys.argv[5])
#bufType= 'Nitrogen' #sys.argv[6]

#time python simulation.py 200 100 400 1.6e13 99.9 Nitrogen
#time python simulation.py 200 100 400 2e12 99.9 Nitrogen
#time python simulation_EDITS.py 10 100 200 2e12 99.9 Nitrogen

# Takes ~5 min on home computer
#time python simulation.py 10 100 800 1.6e13 99.9 Nitrogen 

# For testing
#time python simulation.py 10 100 200 1.6e13 99.9 Nitrogen # 45 seconds on home computer
# Precalculating the log files brings the time down to 28-33 seconds
#time python simulation_EDITS.py 10 100 200 1.6e13 99.9 Nitrogen
#time python simulation_EDITS.py 100 100 400 1.6e13 99.9 Nitrogen


#time python simulation_EDITS.py 10 100 800 1.6e13 99.9 Nitrogen # About 3 minutes
#time python simulation_EDITS.py 10 100 800 1.6e13 99.9 Nitrogen

time python -u simulation_EDITS.py 200 100 400 2e12 99.9 Nitrogen
time python -u simulation_EDITS.py 200 100 400 1.6e13 99.9 Nitrogen
time python -u simulation_EDITS.py 100 100 800 1.6e13 99.9 Nitrogen
