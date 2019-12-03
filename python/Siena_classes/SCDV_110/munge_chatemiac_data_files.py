from pyexcel_xlsx import get_data,save_data
import sys

infilenames = sys.argv[1:]

all = []

header = []
for i,infilename in enumerate(infilenames):
    print("-----------------------------")
    print(infilename.split('/')[-1].split()[0:2])
    print("-----------------------------")

    data = get_data(infilename)

    print(data.keys())
    #exit()
    #print(len(data['Documents']))
    #sheet = data['Documents']
    print(len(data['Ledger']))
    sheet = data['Ledger']

    for j,row in enumerate(sheet):
        if len(row)>0:
            1
            #print(row)
            print(len(row))

        #'''
        if i==0 and j==0:
            all.append(row)

        if j>0:
            all.append(row)
        #'''




#save_data('all_chatemiac_data.xlsx',all)
save_data('all_chatemiac_data_LEDGER.xlsx',all)
