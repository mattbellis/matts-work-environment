#!/usr/bin/env python

import os
from ROOT import *

class ntuplereader:
    def __init__(self,files,branch,varsToUse):
        self.ntupleFiles = files #["sp1228_ddv3NoNNCut.root"]
        tmpClassName = "___tmpClass"
        self.varsToActivate = varsToUse

        if os.path.exists(tmpClassName+".C"):
            os.remove(tmpClassName+".C")
        if os.path.exists(tmpClassName+".h"):
            os.remove(tmpClassName+".h")
        if os.path.exists("getClassValuesxkcd.py"):
            os.remove("getClassValuesxkcd.py")
        if os.path.exists("getClassValuesxkcd.pyc"):
            os.remove("getClassValuesxkcd.pyc")

        self.chain = TChain(branch)
        for i in range(0,len(self.ntupleFiles)):
            self.chain.Add(self.ntupleFiles[i])
        self.chain.MakeClass(tmpClassName)

        f=open(tmpClassName + '.h', 'r')
        self.variableList = []
        for line in f:
            if(line.find("fChain->SetBranchAddress")>-1):
                substring = line[line.find('"')+1:]
                substring = substring[:substring.find('"')]
                self.variableList.append(substring)
        f.close()
        #Delete the file
        
        self.chain = TChain(branch)
        for i in range(0,len(self.ntupleFiles)):
            self.chain.Add(self.ntupleFiles[i])
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

        if os.path.exists(tmpClassName+".C"):
            os.remove(tmpClassName+".C")
        if os.path.exists(tmpClassName+".h"):
            os.remove(tmpClassName+".h")        

    def __del__(self):

        if os.path.exists("getClassValuesxkcd.py"):
            os.remove("getClassValuesxkcd.py")
        if os.path.exists("getClassValuesxkcd.pyc"):
            os.remove("getClassValuesxkcd.pyc")
        if os.path.exists(tmpClassName+".C"):
            os.remove(tmpClassName+".C")
        if os.path.exists(tmpClassName+".h"):
            os.remove(tmpClassName+".h")


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
            print "ERROR: Variable not in list. Returning -99999"
            return -99999

    def activate(self,variableName):
        if variableName in self.variableList:
            self.chain.SetBranchStatus(variableName,1)
        else:
            print "ERROR: Cannot activate variable since it is not in the ntuple"

    def deactivate(self,variableName):
        if variableName in self.variableList:
            self.chain.SetBranchStatus(variableName,0)
        else:
            print "ERROR: Cannot deactivate variable since it is not in the ntuple"
