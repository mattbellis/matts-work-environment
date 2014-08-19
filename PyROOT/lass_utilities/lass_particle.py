from array import array
from math import *

from ROOT import TTree, TMath, TLorentzVector, TVector3

MASSES = [0.0, 0.000511, 0.10566, 0.13957, 0.49368, 0.93827]

###############################################################################
# LassParticle class 
###############################################################################

class LassParticle: 

    '''Helper class to interact with the TTree's dumped out
    by BtaTupleMaker.
    This class defines either a charged final state particle 
    (electron, muon, pion, kaon, proton) or a composite particle
    reconstructed by SimpleComposition.'''
    _P4 = {} # Dictionary of the 4vecs in different refernce frames.
             # This holds TLorentzVectors.

    # Vertex, topology, and beam information for this particle.
    _Topology = -1 
    _VertexIndex = -1 
    _Vertex = TVector3() 
    _BeamIndex = -1 

    # Signifies whether or not this is a composite particle.
    _isComposite = False 

    _lund = 0
    _trk_index = -1

    _name = 'X'

    # Possible names for final state particles.
    _particle_names = ['gamma', 'e', 'mu', 'pi', 'K', 'p']
    _frames = [ '', 'CM' ]
  
    _ndau = []
    _daughter_particle_indices = []

    ###########################################################################
    ###########################################################################

    def __init__(self, name='X', lund=0, P4={}, Vertex=TVector3(), 
                 isComposite=False):
        '''Constructor for LassParticle() object'''

        self._name = name
        self._P4 = P4
        self._Vertex = Vertex
        self._isComposite = isComposite


    ###########################################################################
    ###########################################################################

    def Lund(self): # 
        '''Return the Lund ID of the \'n\'th occurance of \'particle\'.'''
        return self._pvals_int['Lund']

    ###########################################################################
    ###########################################################################

    def GetFloatValue(self, val_name=''): # 
        '''Return some float quantity value.'''
        return self._pvals_float[val_name]

    ###########################################################################
    ###########################################################################

    def TrackIndex(self): # 
        '''Return the Track Index as defined by TrkIdx.'''
        return self._pvals_int['TrkIdx']

    ###########################################################################
    ###########################################################################

    def P4(self, frame=''): # 
        '''Return a TLorentzVector for this particle in the \'frame\' reference frame.
        Possible values for \'frame\' are:
        \'\': Lab frame
        \'CM\': Center of momentum frame for the initial state e+\e- system.'''
        ret = TLorentzVector()
        if frame in self._P4:
            ret = self._P4[frame]

        return ret

    ###########################################################################
    ###########################################################################

    def Vertex(self): # 
        '''Return a TVector3 for this particle.'''
        ret = self._Vertex

        return ret

    ###########################################################################
    ###########################################################################

    def IsPIDSelectorSet(self, selector=-1): # 
        '''Is the \'selector\' (int) set.
        Returns True or False.'''
        self._pid_selector.SetBits( self._pid_bits )
        return self._pid_selector.IsBitSet(selector)

    ###########################################################################
    ###########################################################################

