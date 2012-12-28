from array import array
from math import *
from ROOT import TTree
from ROOT import TMath
from ROOT import TLorentzVector
from ROOT import TVector3
from myPIDselector import *

MASSES = [0.0, 0.000511, 0.10566, 0.13957, 0.49368, 0.93827]

######### BaBarParticle class #################
class BaBarParticle: 
  '''Helper class to interact with the TTree's dumped out
  by BtaTupleMaker.
  This class defines either a charged final state particle 
  (electron, muon, pion, kaon, proton) or a composite particle
  reconstructed by SimpleComposition.'''
  _P4 = {} # Dictionary of the 4vecs in different refernce frames.
                               # Should hold TLorentzVectors

  _Vertex = TVector3() # Vertex, right now from Vtx

  _isComposite = False # Signifies whether or not this is a 
                       # composite particle.

  _lund = 0
  _p_index = -1

  _pvals_int = {}
  _pvals_float = {}

  _name = 'X'

  # For final state particles
  _particle_names = ['gamma', 'e', 'mu', 'pi', 'K', 'p']
  _frames = [ '', 'CM' ]
  
  _stored_value_names = ['TrkIdx', 'Lund', 'SelectorsMap']
  _pval_integers = []

  _ndau = []
  _daughter_particle_indices = []

  pval_daughter_strings = ['Lund', 'Idx']
  pval_daughter_integers = []

  pidSelectors = []
  _pid_selector = None

  _pid_bits = 0

  MAX_NPART = 64

  ############################################################################
  def __init__(self, name='X', lund=0, P4={}, Vertex=TVector3(), isComposite=False, pvals_float={}, pvals_int={}, pid_bits=0): # 
    '''Constructor for BaBarParticle() object'''

    self._name = name
    self._P4 = P4
    self._Vertex = Vertex
    self._isComposite = isComposite
    self._pvals_float = pvals_float
    self._pvals_int = pvals_int
    self._pid_bits = pid_bits
    self._pid_selector = PIDselector(self._name)


  ##############################################################################
  #############################################################################
  def Lund(self): # 
    '''Return the Lund ID of the \'n\'th occurance of \'particle\'.'''
    return self._pvals_int['Lund']

  ##############################################################################
  #############################################################################
  def GetFloatValue(self, val_name=''): # 
    '''Return some float quantity value.'''
    return self._pvals_float[val_name]

  ##############################################################################
  ##############################################################################
  #############################################################################
  def TrackIndex(self): # 
    '''Return the Track Index as defined by TrkIdx.'''
    return self._pvals_int['TrkIdx']

  ##############################################################################
  ##############################################################################
  ##############################################################################
  ##############################################################################
  def P4(self, frame=''): # 
    '''Return a TLorentzVector for this particle in the \'frame\' reference frame.
    Possible values for \'frame\' are:
    \'\': Lab frame
    \'CM\': Center of momentum frame for the initial state e+\e- system.'''
    ret = TLorentzVector()
    if frame in self._P4:
      ret = self._P4[frame]

    return ret

  ##############################################################################
  def Vertex(self): # 
    '''Return a TVector3 for this particle.'''
    ret = self._Vertex

    return ret

  ##############################################################################
  def IsPIDSelectorSet(self, selector=-1): # 
    '''Is the \'selector\' (int) set.
    Returns True or False.'''
    self._pid_selector.SetBits( self._pid_bits )
    return self._pid_selector.IsBitSet(selector)


  ##############################################################################
  ##############################################################################
