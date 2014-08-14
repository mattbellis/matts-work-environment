## Translated from 'lineset_test.C'.
## Run as: python -i lineset_test.py

import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

def lineset_test(nlines = 40, nmarkers = 4):
    r = ROOT.TRandom(0)
    s = 100

    ls = ROOT.TEveStraightLineSet()

    for j in range(0,1000):
      for i in range(nlines):
          ls.AddLine( r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s) ,
                      r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s))
          nm = int(nmarkers*r.Rndm())
          for m in range(nm):
              ls.AddMarker( i, r.Rndm() )
      ls.SetMarkerSize(1.5)
      ls.SetMarkerStyle(4)

      ROOT.gEve.AddElement(ls)
      ROOT.gEve.Redraw3D()
      return ls

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    lineset_test()
