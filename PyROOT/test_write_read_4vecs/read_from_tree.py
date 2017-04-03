import ROOT 

# Open the file
f = ROOT.TFile("myrootfile.root")
f.ls() # Print out what's in it.

# Pull out the tree
tree = f.Get("T")
tree.Print() # Print what branches it has

# Event loop
nev = tree.GetEntries()

for n in range (nev):
    tree.GetEntry(n)

    print(tree.nmuon)

    for i in range(tree.nmuon):
        print(tree.muonpt[i])


