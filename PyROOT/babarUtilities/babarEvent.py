from array import array
from math import *
from ROOT import TTree
from ROOT import TMath
from ROOT import TLorentzVector
from myPIDselector import *
#import babarParticle
from babarParticle import *

MASSES = [0.0, 0.000511, 0.10566, 0.13957, 0.49368, 0.93827]

######### BaBarEvent class #################
class BaBarEvent: 
  '''Helper class to interact with the TTree's dumped out
  by BtaTupleMaker.'''
  _tree = TTree()
  #_beamP4 = TLorentzVector() # P4 of intial e+e- system. 

  # Place holders
  # For beam
  beamstrings = [ 'eePx', 'eePy', 'eePz', 'eeE' ]
  _beam4vec = [ array('f', [0.0]), array('f', [0.0]), array('f', [0.0]), array('f', [0.0]) ] 

  # For final state particles
  particle_names = ['gamma', 'e', 'mu', 'pi', 'K', 'p']
  pvecstring = ['energy', 'p3', 'costh', 'phi']
  vvecstring = ['Vtxx', 'Vtxy', 'Vtxz']
  frames = [ '', 'CM' ]
  num_particle = []
  particle4vec  = []
  particlevvec  = []

  pval_integer_strings = ['TrkIdx', 'Lund', 'SelectorsMap']
  pval_integers = []

  pval_float_strings = ['FlightLen', 'postFitMass']
  pval_floats = []

  _particle_integers_strings = ['TrkIdx', 'Lund', 'SelectorsMap']
  _particle_integers = {}

  _stored_float_values = {}

  _p_ndau = []
  pval_daughter_strings = ['Lund', 'Idx']
  pval_daughter_integers = []

  pidSelectors = []

  MAX_NPART = 64
  FIRST_COMPOSITE_INDEX = 6

  ############################################################################
  ############################################################################
  ############################################################################
  def __init__(self, tree=None, composite_particles=[]): # TTree object
    '''Constructor for BaBarEvent() object'''

    self._tree = tree
    #self._beamP4 = TLorentzVector()

    ############################################
    # Push on any composite particles.
    ############################################
    for cp in composite_particles:
      self.particle_names.append(cp)

    # Set the addresses for the beam info
    for i,s in enumerate(self.beamstrings):
      self._tree.SetBranchAddress( s, self._beam4vec[i] )

    # Search for the particles based on if n is in there
    for i in range(0, len(self.particle_names)):
      self.num_particle.append( array('i', [0]) )
      if i < 6:
        if i==0:
          self.pidSelectors.append("e")
        else:
          print "appending %s" % (self.particle_names[i])
          self.pidSelectors.append(PIDselector(self.particle_names[i]))

      # 4vec info
      dum = []
      for n,f in enumerate(self.frames):
        dum.append([])
        for k in range(0,4):
          dum[n].append(array('f', [0.0])*self.MAX_NPART )
      self.particle4vec.append( dum )
      
      # Vertex vec info
      dum = []
      for k in range(0,3):
        dum.append(array('f', [0.0])*self.MAX_NPART )
      self.particlevvec.append( dum )
      
      # Integers
      dum = []
      nvals = len(self.pval_integer_strings)
      for k in range(0,nvals):
        dum.append(array('i', [0])*self.MAX_NPART )
      self.pval_integers.append( dum )
      
      # Floats
      dum = []
      nvals = len(self.pval_float_strings)
      for k in range(0,nvals):
        dum.append(array('f', [0])*self.MAX_NPART )
      self.pval_floats.append( dum )

      # Daughter
      self._p_ndau.append(0)
      dum = []
      nvals = len(self.pval_daughter_strings)
      for n in range(0,16): # Number of possible daughters
        dum.append([])
        for k in range(0,nvals):
          dum[n].append(array('i', [0])*self.MAX_NPART )
      self.pval_daughter_integers.append( dum )
      

    #############################################
    # Now set the values with the TTree....
    # ...if they exist
    #############################################
    branch_list = self._tree.GetListOfBranches()

    #############################################
    # First do this for the particles
    #############################################
    for j,p in enumerate(self.particle_names):
      name = "n%s" % ( p )
      if branch_list.FindObject(name):
        self._tree.SetBranchAddress( name, self.num_particle[j] )
        for nf,frame in enumerate(self.frames):
          for i,s in enumerate(self.pvecstring):
            name = "%s%s%s" % ( p, s, frame )
            print name
            self._tree.SetBranchAddress( name, self.particle4vec[j][nf][i] )

        # Vertex vec
        for i,s in enumerate(self.vvecstring):
          name = "%s%s" % ( p, s )
          print name
          self._tree.SetBranchAddress( name, self.particlevvec[j][i] )

        ########### Ints ################
        for i,s in enumerate(self.pval_integer_strings):
          name = "%s%s" % ( p, s )
          print name
          self._tree.SetBranchAddress( name, self.pval_integers[j][i] )

        ########### Ints ################
        for i,s in enumerate(self.pval_float_strings):
          name = "%s%s" % ( p, s )
          print name
          self._tree.SetBranchAddress( name, self.pval_floats[j][i] )

        # Search for daughter info
        for n in range(0,16):
          name = "%sd%dLund" % ( p, n )
          if branch_list.FindObject(name):
            print name
            self._p_ndau[j] += 1
            for i,s in enumerate(self.pval_daughter_strings):
              name = "%sd%d%s" % ( p, n, s )
              print name
              self._tree.SetBranchAddress( name, self.pval_daughter_integers[j][n][i] )

      else:
        print "Cannot find particle %s!!!!!!!!" % (p)
        

  ##############################################################################
  def GetEvent(self, nev=0): # 
    '''Initialize the BaBar event with the \'nev\'th event in the TTree.'''
    self._tree.GetEvent(nev)


  ##############################################################################
  def MyTree(self): # 
    '''Return the TTree object with which the BaBarEvent has been initialized.'''
    return self._tree

  ##############################################################################
  def BeamP4(self): # 
    '''Return a TLorentzVector associated with the sum of the e+/e- beams.'''
    ret = TLorentzVector()
    ret.SetXYZT( self._beam4vec[0][0], self._beam4vec[1][0], self._beam4vec[2][0], self._beam4vec[3][0] )
    return ret

  ##############################################################################
  def npart(self, particle="X"): # 
    '''Return the number of particles of type \'particle\'.'''
    i = self.IsParticleInTree(particle)
    if i >= 0:
      return self.num_particle[i][0]
    else:
      return -1

  ##############################################################################
  def Lund(self, particle="X", n=0): # 
    '''Return the Lund ID of the \'n\'th occurance of \'particle\'.'''
    i = self.IsParticleInTree(particle)
    j = self.IsStringInList( self.pval_integer_strings, 'Lund')
    if i >= 0 and j>=0:
      return self.pval_integers[i][j][n]
    else:
      return 0

  ##############################################################################
  # Beginning of daughters info.
  ##############################################################################
  def NumberOfDaughters(self, particle="X"): # 
    '''Return the number of daughters (decay products) of \'particle\'.'''
    i = self.IsParticleInTree(particle)
    if i >= 0:
      return self._p_ndau[i]
    else:
      return 0

  ##############################################################################
  def DaughterLund(self, particle="X", n=0, ndau=0): # 
    '''Return the Lund ID of the \'ndau\'th daugther of the \'n\'th
    occurance of \'particle\'.'''
    i = self.IsParticleInTree(particle)
    j = self.IsStringInList( self.pval_daughter_strings, 'Lund')
    ndau += 1
    if i >= 0 and j>=0:
      return self.pval_daughter_integers[i][ndau][j][n]
    else:
      return 0

  ##############################################################################
  def DaughterIndex(self, particle="X", n=0, ndau=0): # 
    '''Return the particle index of the \'ndau\'th daugther of the \'n\'th
    occurance of \'particle\'.'''
    i = self.IsParticleInTree(particle)
    j = self.IsStringInList( self.pval_daughter_strings, 'Idx')
    ndau += 1
    if i >= 0 and j>=0:
      return self.pval_daughter_integers[i][ndau][j][n]
    else:
      return 0

  ##############################################################################
  def TrackIndex(self, particle="X", n=0): # 
    '''Return the track index of the \'n\'th occurance of \'particle\'.'''
    i = self.IsParticleInTree(particle)
    j = self.IsStringInList( self.pval_integer_strings, 'TrkIdx')
    if i >= 0 and j>=0:
      return self.pval_integers[i][j][n]
    else:
      return 0

  ##############################################################################
  def SelectorsMap(self, particle="X", n=0): # 
    '''Return the selectors map of the \'n\'th occurance of \'particle\'.
    Note that this does not exist for any composite particles except
    Brem recovered electrons, where this maps onto the electron itself.'''
    i = self.IsParticleInTree(particle)
    j = self.IsStringInList( self.pval_integer_strings, 'SelectorsMap')
    if i >= 0 and j>=0:
      return self.pval_integers[i][j][n]
    else:
      return 0

  ##############################################################################
  def IsPIDSelectorSet(self, particle="X", n=0, selector=-1): # 
    '''Is the \'selector\' (int) set for the \'n\'th occurance 
    of \'particle\'.
    Returns True or False.'''
    i = self.IsParticleInTree(particle)
    if i >= 0:
      self.pidSelectors[i].SetBits( self.SelectorsMap(particle, self.TrackIndex(particle, n) ) )
      #self.pidSelectors[i].PrintSelectors()
      return self.pidSelectors[i].IsBitSet(selector)
    else:
      return False

  ##############################################################################
  def PrintPIDSelectors(self, particle="X", n=0): # 
    '''Prints the PID selectors which are set to True for the 
    \'n\'th occurance of \'particle\'.'''
    i = self.IsParticleInTree(particle)
    if i >= 0:
      self.pidSelectors[i].SetBits( self.SelectorsMap(particle, self.TrackIndex(particle, n) ) )
      self.pidSelectors[i].PrintSelectors()
      return True
    else:
      return False

  ##############################################################################
  ##############################################################################
  ##############################################################################
  ##############################################################################
  def P4(self, particle="X", n=0, frame=''): # 
    '''Return a TLorentzVector for \'n\'th occurance of \'particle\' in the
    \'frame\' reference frame.
    Possible values for \'frame\' are:
    \'\': Lab frame
    \'CM\': Center of momentum frame for the initial state e+\e- system.'''
    ret = TLorentzVector()
    i = self.IsParticleInTree(particle)
    j = self.IsStringInList(self.frames, frame)
    if i>=0:
      energy =     self.particle4vec[i][j][0][n]
      pmag =       self.particle4vec[i][j][1][n]
      theta = acos(self.particle4vec[i][j][2][n])
      phi =        self.particle4vec[i][j][3][n]
      x = pmag * sin(theta) * cos( phi )
      y = pmag * sin(theta) * sin( phi )
      z = pmag * cos(theta)
      ret.SetXYZT( x, y, z, energy)

    return ret

  ##############################################################################
  def Vertex(self, particle="X", n=0): # 
    '''Return a TVector3 for \'n\'th occurance of \'particle\'.'''
    ret = TVector3()
    i = self.IsParticleInTree(particle)
    if i>=0:
      x = self.particlevvec[i][0][n]
      y = self.particlevvec[i][1][n]
      z = self.particlevvec[i][2][n]
      ret.SetXYZ( x, y, z )

    return ret

  ##############################################################################
  ##############################################################################
  def GetParticle(self, particle="X", n=0): #
    '''Return a BaBarParticle of the \'n\'th occurance of \'X\'.'''
    #def __init__(self, lund=0, P4={}, isComposite=False, pvals_float=[], pvals_int=[]): # TTree object
    lund = self.Lund(particle, n)
    isComposite = False
    i = self.particle_names.index( particle )
    if i >= self.FIRST_COMPOSITE_INDEX:
      isComposite = True
    p4s = {}
    for f in self.frames:
      p4s[f] =  self.P4(particle, n, f)

    vertex = self.Vertex(particle, n)

    # Grab the integer values
    vals_int= {}
    for j,v in enumerate(self.pval_integer_strings):
      vals_int[v] = self.pval_integers[i][j][n]

    # Grab the float values
    vals_float= {}
    for j,v in enumerate(self.pval_float_strings):
      vals_float[v] = self.pval_floats[i][j][n]

    pid_bits = self.SelectorsMap(particle, self.TrackIndex(particle, n) ) 

    ret = BaBarParticle(particle, lund, p4s, vertex, isComposite, vals_float, vals_int, pid_bits)
    return ret


  ##############################################################################
  def IsParticleInTree(self, particle="X"):
    '''Check to see if \'particle\' is in the TTree object passed in to this BaBarEvent.'''
    i = -1
    try:
      i = self.particle_names.index( particle )
    except ValueError:
      print "Cannot find particle %s in this file/tree!!!!!" % (particle)
      i = -1 # no match

    return i

  ##############################################################################
  ##############################################################################
  def IsStringInList(self, mylist=[], mystring="X"):
    '''Check to see if \'mystring\' is in any of the lists in this BaBarEvent.
    This is a helper function for some internal checks.'''
    i = -1
    try:
      i = mylist.index( mystring )
    except ValueError:
      print "Cannot find string %s in this list!!!!!" % (mystring)
      i = -1 # no match

    return i

  ##############################################################################
