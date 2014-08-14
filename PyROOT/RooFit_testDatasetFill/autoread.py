#!/usr/bin/env python

import os
from ROOT import *

class ntuplereader:
    def __init__(self,files,branch,varsToUse):
        self.ntupleFiles = files #["sp1228_ddv3NoNNCut.root"]

        self.varsToActivate = varsToUse

        if os.path.exists("getClassValuesxkcd.py"):
            os.remove("getClassValuesxkcd.py")
        if os.path.exists("getClassValuesxkcd.pyc"):
            os.remove("getClassValuesxkcd.pyc")

        self.chain = TChain(branch)
        for i in range(0,len(self.ntupleFiles)):
            self.chain.Add(self.ntupleFiles[i])

        self.variableList = []
        branch_list = self.chain.GetListOfBranches();
        nbranches = branch_list.GetEntries();
        for br in range(0, nbranches):
            br_name = branch_list[br].GetName()
            self.variableList.append(br_name)

        self.chain.SetBranchStatus("*",0)

        for var in self.varsToActivate:
            if var in self.variableList:
                self.chain.SetBranchStatus(var,1)
        
        f=open("getClassValuesxkcd.py",'w')
        
        f.write('def getValue(varname,chain):\n')
        for varname in self.variableList:
            f.write('\tif(varname == "' + varname + '"):\n')
            f.write('\t\treturn chain.' + varname + "\n")
        f.close()
        
        import getClassValuesxkcd
        
        self.chain.GetEntry(0)

    def __del__(self):

        if os.path.exists("getClassValuesxkcd.py"):
            os.remove("getClassValuesxkcd.py")
        if os.path.exists("getClassValuesxkcd.pyc"):
            os.remove("getClassValuesxkcd.pyc")

    def getChain(self):
        return self.chain

    def getEntries(self):
        return self.chain.GetEntries()

    def getVariables(self):
        return self.variableList
    
    def entry(self,index):
        self.chain.GetEntry(index)
    
    def get(self,variableName):
        import getClassValuesxkcd
        if variableName in self.variableList and variableName in self.varsToActivate:
            return getClassValuesxkcd.getValue(variableName,self.chain)
        else:
            print "ERROR: Variable", variableName, "not present in ntuples"
            quit()
            return -99999

    def activate(self,variableName):
        if variableName in self.variableList:
            self.chain.SetBranchStatus(variableName,1)
        else:
            print "ERROR: Cannot activate variable", variableName, "since it is not in the ntuple"
            quit()

    def deactivate(self,variableName):
        if variableName in self.variableList:
            self.chain.SetBranchStatus(variableName,0)
        else:
            print "ERROR: Cannot deactivate variable", variableName, "since it is not in the ntuple"
            quit()
