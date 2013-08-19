from array import array
class PIDselector:
  ''' PIDselector function.
  A helper function to more easily test to see if 
  a PID selector is set for a given track/particle.'''
  # Always remember the *self* argument
  bits = []
  def __init__(self,particle=None):
    self.particle = particle or []
    self.max_bits = 0
    self.bits = []
    self.selectors = []
    for i in range(0,32):
      self.bits.append(0)
      self.selectors.append(" ")
    # Set the particle info
    if self.particle == "p" or self.particle == "proton":
      self.max_bits = 18 
      self.selectors[0] =  "VeryLooseLHProtonSelection"
      self.selectors[1] =  "LooseLHProtonSelection"
      self.selectors[2] =  "TightLHProtonSelection"
      self.selectors[3] =  "VeryTightLHProtonSelection"

      self.selectors[4] =  "VeryLooseGLHProtonSelection"
      self.selectors[5] =  "LooseGLHProtonSelection"
      self.selectors[6] =  "TightGLHProtonSelection"
      self.selectors[7] =  "VeryTightGLHProtonSelection"
      
      self.selectors[8] =  "VeryLooseELHProtonSelection"
      self.selectors[9] =  "LooseELHProtonSelection"
      self.selectors[10] =  "TightELHProtonSelection"
      self.selectors[11] =  "VeryTightELHProtonSelection"

      self.selectors[12] =  "SuperLooseKMProtonSelection"
      self.selectors[13] =  "VeryLooseKMProtonSelection"
      self.selectors[14] =  "LooseKMProtonSelection"
      self.selectors[15] =  "TightKMProtonSelection"
      self.selectors[16] =  "VeryTightKMProtonSelection"
      self.selectors[17] =  "SuperTightKMProtonSelection"
      
    elif self.particle=="K" or self.particle == "k" or self.particle == "kaon":
      self.max_bits = 30 
      self.selectors[0] =  "NotPionKaonMicroSelection"
      self.selectors[1] =  "VeryLooseKaonMicroSelection"
      self.selectors[2] =  "LooseKaonMicroSelection"
      self.selectors[3] =  "TightKaonMicroSelection"
      self.selectors[4] =  "VeryTightKaonMicroSelection"

      self.selectors[5] =  "NotPionNNKaonMicroSelection"
      self.selectors[6] =  "VeryLooseNNKaonMicroSelection"
      self.selectors[7] =  "LooseNNKaonMicroSelection"
      self.selectors[8] =  "TightNNKaonMicroSelection"
      self.selectors[9] =  "VeryTightNNKaonMicroSelection"

      self.selectors[11] =  "NotPionLHKaonMicroSelection"
      self.selectors[12] =  "VeryLooseLHKaonMicroSelection"
      self.selectors[13] =  "LooseLHKaonMicroSelection"
      self.selectors[14] =  "TightLHKaonMicroSelection"
      self.selectors[15] =  "VeryTightLHKaonMicroSelection"

      self.selectors[16] =  "VeryLooseGLHKaonMicroSelection"
      self.selectors[17] =  "LooseGLHKaonMicroSelection"
      self.selectors[18] =  "TightGLHKaonMicroSelection"
      self.selectors[19] =  "VeryTightGLHKaonMicroSelection"

      self.selectors[20] =  "NotPionBDTKaonMicroSelection"
      self.selectors[21] =  "VeryLooseBDTKaonMicroSelection"
      self.selectors[22] =  "LooseBDTKaonMicroSelection"
      self.selectors[23] =  "TightBDTKaonMicroSelection"
      self.selectors[24] =  "VeryTightBDTKaonMicroSelection"
         
      self.selectors[25] =  "SuperLooseKMKaonMicroSelection"
      self.selectors[26] =  "VeryLooseKMKaonMicroSelection"
      self.selectors[27] =  "LooseKMKaonMicroSelection"
      self.selectors[28] =  "TightKMKaonMicroSelection"
      self.selectors[29] =  "VeryTightKMKaonMicroSelection"
      self.selectors[30] =  "SuperTightKMKaonMicroSelection"

    elif self.particle == "pi" or self.particle == "pion":
      self.max_bits = 16 
      self.selectors[0] =  "PidRoyPionSelectionLoose"
      self.selectors[1] =  "PidRoyPionSelectionNotKaon"

      self.selectors[2] =  "VeryLooseLHPionSelection"
      self.selectors[3] =  "LooseLHPionSelection"
      self.selectors[4] =  "TightLHPionSelection"
      self.selectors[5] =  "VeryTightLHPionSelection"

      self.selectors[6] =  "VeryLooseGLHPionSelection"
      self.selectors[7] =  "LooseGLHPionSelection"
      self.selectors[8] =  "TightGLHPionSelection"
      self.selectors[9] =  "VeryTightGLHPionSelection"
        
      self.selectors[10] =  "SuperLooseKMPionSelection"
      self.selectors[11] =  "VeryLooseKMPionSelection"
      self.selectors[12] =  "LooseKMPionSelectionLooseGLHPionSelection"
      self.selectors[13] =  "TightKMPionSelection"
      self.selectors[14] =  "VeryTightKMPionSelection"
      self.selectors[15] =  "SuperTightKMPionSelection"
    elif  self.particle == "e" or self.particle == "electron":
      self.max_bits = 12 
      self.selectors[0] =  "NoCalElectronMicroSelection"
      self.selectors[1] =  "VeryLooseElectronMicroSelection"
      self.selectors[2] =  "LooseElectronMicroSelection"
      self.selectors[3] =  "TightElectronMicroSelection"
      self.selectors[4] =  "VeryTightElectronMicroSelection"

      self.selectors[5] =  "PidLHElectrons"

      self.selectors[6] =  "SuperLooseKMElectronMicroSelection"
      self.selectors[7] =  "VeryLooseKMElectronMicroSelection"
      self.selectors[8] =  "LooseKMElectronMicroSelection"
      self.selectors[9] =  "TightKMElectronMicroSelection"
      self.selectors[10] =  "VeryTightKMElectronMicroSelection"
      self.selectors[11] =  "SuperTightKMElectronMicroSelection"
    
    elif self.particle == "mu" or self.particle == "muon":
      self.max_bits = 26 
         
      self.selectors[0] =  "MinimumIoniziongMuonMicroSelection"
      self.selectors[1] =  "VeryLooseMuonMicroSelection"
      self.selectors[2] =  "LooseMuonMicroSelection"
      self.selectors[3] =  "TightMuonMicroSelection"
      self.selectors[4] =  "VeryTightMuonMicroSelection"
        
      self.selectors[5] =  "NNVeryLooseMuonSelection"
      self.selectors[6] =  "NNLooseMuonSelection"
      self.selectors[7] =  "NNTightMuonSelection"
      self.selectors[8] =  "NNVeryTightMuonSelection"
      self.selectors[9] =  "NNVeryLooseMuonSelectionFakeRate"
      self.selectors[10] =  "NNLooseMuonSelectionFakeRate"
      self.selectors[11] =  "NNTightMuonSelectionFakeRate"
      self.selectors[12] =  "NNVeryTightMuonSelectionFakeRate"

      self.selectors[13] =  "LikeVeryLooseMuonSelection"
      self.selectors[14] =  "LikeLooseMuonSelection"
      self.selectors[15] =  "LikeTightMuonSelection"
        
      self.selectors[16] =  "BDTVeryLooseMuonSelection"
      self.selectors[17] =  "BDTLooseMuonSelection"
      self.selectors[18] =  "BDTTightMuonSelection"
      self.selectors[19] =  "BDTVeryTightMuonSelection"
      self.selectors[20] =  "BDTVeryLooseMuonSelectionFakeRate"
      self.selectors[21] =  "BDTLooseMuonSelectionFakeRate"
      self.selectors[22] =  "BDTTightMuonSelectionFakeRate"
      self.selectors[23] =  "BDTVeryTightMuonSelectionFakeRate"

      self.selectors[24] =  "BDTLoPLooseMuonSelection"
      self.selectors[25] =  "BDTLoPTightMuonSelection "

  ##############################################
  def SetBits(self, val=0 ):
    '''Set the bits for the PID selctors by passing in \'val\'.
    This is the number in the XXXSelectorMap.'''
    for i in range(0,32):
      self.bits[i] = 0

    if val > 2.0**(self.max_bits):
      print "WARNING: value is set higher than 2^(max_bits-1)!!!!"
      print val
      print self.max_bits
      print 2.0**(self.max_bits)

    digit = 1
    for i in range(0,self.max_bits):
      #digit = int(2.0**(i+1))
      digit *= 2
      test = val%digit
      if test:
        self.bits[i] = 1
      else:
        self.bits[i] = 0
      val -= test

  ##############################################
  def IsBitSet(self, val):
    '''Check if selector number \'val\' is set.
    Returns True or False.'''
    if(self.bits[val]):
      return True
    return False

  ##############################################
  def IsSelectorSet(self, selector):
    '''Check to see if a particular selector is set, based
    on the string \'selector\'.
    Returns True or False.'''
    for i in range(0,self.max_bits):
      if self.selectors[i] == selector:
        if self.bits[i]:
          return True
        return False
    print "Warning!!!!  " + self.selector + " is not a valid choice for particle " + self.particle
    return False

  ##############################################
  def PrintBits(self):
    '''Print to the screen the bit selectors that are set 
    to True.'''
    output = "bits: "
    for i in range(0,self.max_bits):
      output += str(self.bits[i])
    print output

  ##############################################
  def PrintSelectors(self):
    '''Print to the screen the names of the selectors that are set 
    to True.'''
    output = "selectors:\n"
    for i in range(0,self.max_bits):
      if self.bits[i]:
        output += "\t" + str(self.selectors[i]) + "\n" 
      else:
        output += "\t" + " __ " + "\n"
    print output

