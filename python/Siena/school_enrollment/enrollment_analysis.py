# Use the tablula *local* server (described in the tablua README.txt)
# to extract the table as a .tsv file
import sys
import numpy as np
import matplotlib.pylab as plt

################################################################################
def parse_year(number_string):

    app,acc,con,y = None,None,None,None
    newvals = number_string.split()
    print(newvals)
    if len(newvals)>=3 and newvals[0] != 'Applied' :
        app = int(number_string.split()[0].replace(',',''))
        acc = int(number_string.split()[1].replace(',',''))
        con = int(number_string.split()[2].replace(',',''))
        y = 0
        if acc > 0:
            y = con/acc

    return app,acc,con,y
################################################################################
################################################################################
def extract_values(mydict,majors):

    # If more than one major is passed in (as a list), the total is
    # returned for all those majors

    if type(majors) != list:
        majors = [majors]

    values = {}
    for key in ['applied','accepted','confirmed','yield']:
        values[key] = []

    if len(majors)==1:
        major = majors[0]
        for key in ['applied','accepted','confirmed','yield']:
            values[key].append(mydict[major]['prior2'][key])
            values[key].append(mydict[major]['prior'][key])
            values[key].append(mydict[major]['current'][key])
    else:
        for i,major in enumerate(majors):
            if i==0:
                for key in ['applied','accepted','confirmed']:
                    values[key].append(0)
                    values[key].append(0)
                    values[key].append(0)

            for key in ['applied','accepted','confirmed']:
                print(values)
                print(values[key])
                values[key][0] += mydict[major]['prior2'][key]
                values[key][1] += mydict[major]['prior'][key]
                values[key][2] += mydict[major]['current'][key]

        values['yield'].append(values['confirmed'][0]/values['accepted'][0])
        values['yield'].append(values['confirmed'][1]/values['accepted'][1])
        values['yield'].append(values['confirmed'][2]/values['accepted'][2])


    return values

################################################################################


# Get the data
infilename = sys.argv[1]

data = {}

for line in open(infilename,'r'):
    vals = line.split('\t')
    if len(vals)<6:
        continue

    key = vals[0].strip()
    print(key)
    #print(vals)
    app,acc,con,y = parse_year(vals[1])
    if app is not None:
        data[key] = {}
        data[key]['current'] = {}
        data[key]['current']['applied'] = app
        data[key]['current']['accepted'] = acc
        data[key]['current']['confirmed'] = con
        data[key]['current']['yield'] = y

    app,acc,con,y = parse_year(vals[3])
    if app is not None:
        data[key]['prior'] = {}
        data[key]['prior']['applied'] = app
        data[key]['prior']['accepted'] = acc
        data[key]['prior']['confirmed'] = con
        data[key]['prior']['yield'] = y

    app,acc,con,y = parse_year(vals[5])
    if app is not None:
        data[key]['prior2'] = {}
        data[key]['prior2']['applied'] = app
        data[key]['prior2']['accepted'] = acc
        data[key]['prior2']['confirmed'] = con
        data[key]['prior2']['yield'] = y

'''
for d in data.keys():
    print(d)
    print(data[d])
'''

majors = ['App Physics-Electrical',
        'App Physics-Mechanical',
        'Applied Physics',
        'Engineering 3/2 Program',
        'Engineering 4/1 Program',
        'Physics',
        'Physics BioChem',
        'Computational Physics',
        'Physics Education',
        'Physics ThreeTwo' ]

groups1 = ['App Physics-Electrical',
        'App Physics-Mechanical',
        'Applied Physics',
        'Engineering 3/2 Program',
        'Engineering 4/1 Program',
        'Physics ThreeTwo' ]


groups2 = [
        'Physics',
        'Physics BioChem',
        'Computational Physics',
        'Physics Education',
         ]


xticks = ['2019','2020','2021']
xpts = [1,2,3]

################################################################################
plt.figure(figsize=(12,5))
for major in majors:
    y = extract_values(data,major)
    plt.plot(xticks,y['applied'],'o-',label=major)
    plt.plot(xticks,y['accepted'],'v-')
    plt.plot(xticks,y['confirmed'],'s-')
plt.legend()
plt.xlim(0,3)
plt.savefig('fig1.png')

################################################################################
plt.figure(figsize=(12,5))
for major in groups1:
    y = extract_values(data,major)
    plt.plot(xticks,y['accepted'],'o-',label=major)
plt.legend()
plt.xlim(0,3)
plt.savefig('fig2.png')

################################################################################
plt.figure(figsize=(12,5))
for major in groups2:
    y = extract_values(data,major)
    plt.plot(xticks,y['accepted'],'o-',label=major)
plt.legend()
plt.xlim(0,3)
plt.savefig('fig3.png')


################################################################################
for key in ['applied','accepted','confirmed']:
    plt.figure(figsize=(12,5))
    y1 = extract_values(data,groups1)
    y2 = extract_values(data,groups2)
    plt.plot(xticks,y2[key],'o-',label='Physics, Bio, Comp, Ed')
    plt.plot(xticks,y1[key],'o-',label='Applied et al')
    plt.legend()
    plt.title(key)
    plt.xlim(0,3)
    plt.savefig('fig4'+key+'.png')


################################################################################
plt.figure(figsize=(12,5))
y1 = extract_values(data,groups1)
y2 = extract_values(data,groups2)
plt.plot(xticks,y2['applied'],'o-',markersize=15,label='APPLIED - Physics, Bio, Comp, Ed')
plt.plot(xticks,y1['applied'],'o-',markersize=15,label='APPLIED - Applied et al')
plt.plot(xticks,y2['accepted'],'v-',markersize=15,label='ACCEPTED - Physics, Bio, Comp, Ed')
plt.plot(xticks,y1['accepted'],'v-',markersize=15,label='ACCEPTED - Applied et al')
plt.legend()
plt.xlim(0,3)
plt.savefig('fig5.png')




plt.show()
