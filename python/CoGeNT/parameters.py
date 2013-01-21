############################################################################
# Declare the ranges.
############################################################################

def fitting_parameters(flag=0):
    #ranges = [[0.5,3.2],[1.0,459.0]]
    ##dead_days = [[68,74], [102,107],[306,308]]
    #subranges = [[],[[1,68],[75,102],[108,306],[309,459]]]

    #ranges = [[0.5,2.0],[108.0,917.0]]
    #subranges = [[],[[108,306],[309,459],[551,917]]]

    #ranges = [[0.5,3.5],[108.0,917.0]]
    #subranges = [[],[[108,306],[309,459],[551,917]]]

    #ranges = [[0.5,3.5],[551.0,917.0]]
    #subranges = [[],[[551,917]]]

    #ranges = [[0.5,3.5],[1.0,917.0]]
    #subranges = [[],[[1,68],[75,102],[108,306],[309,459],[551,917]]]

    ranges = [[0.5,3.0],[1.0,917.0]]
    subranges = [[],[[1,68],[75,102],[108,306],[309,459],[551,917]]]

    #ranges = [[0.5,3.0],[100.0,917.0]]
    #subranges = [[],[[100,102],[108,306],[309,459],[551,917]]]

    #ranges = [[0.5,3.2],[1.0,917.0]]
    #ranges = [[0.5,3.5],[1.0,917.0]]
    #ranges = [[0.5,3.5],[1.0,917.0]]
    #ranges = [[0.5,3.5],[1.0,459.0]]
    #dead_days = [[68,74], [102,107],[306,308]]
    #subranges = [[],[[1,68],[75,102],[108,306],[309,459],[551,917]]]
    #subranges = [[],[[1,68],[75,102],[108,306],[309,459]]]
    if flag==5 or flag==6:
        subranges = [[],[[1,917]]]

    #nbins = [108,30]
    #nbins = [150,30]
    nbins = [75,50]
    #nbins = [150,15]
    #nbins = [200,30]
    #nbins = [100,30]

    if ranges[1][1]==459:
        nbins[1] = 15

    return ranges,subranges,nbins




