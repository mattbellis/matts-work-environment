## Translated from 'lineset_test.C'.
## Run as: python -i lineset_test.py

import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

def hello_eve(line):
    r = ROOT.TRandom(0)
    s = 100

    ls = ROOT.TEveStraightLineSet()

    vars = line.split()
    num_f = len(vars)/5 - 1
    # start with the first final state particle
    for k in range(1, num_f+1): 
      E = float(vars[k*5 + 1])
      x = float(vars[k*5 + 2])
      y = float(vars[k*5 + 3])
      z = float(vars[k*5 + 4])

      print x

      ls.AddLine( 0, 0, 0, x, y, z )

    ROOT.gEve.AddElement(ls)
    ROOT.gEve.Redraw3D()

    ls.DestroyElements()

    return ls



if __name__=='__main__':
  ROOT.PyGUIThread.finishSchedule()

  filename = "exercise_1.txt"
  file = open(filename)

  count = 0
  for line in file:
    count += 1
    if count > 10:
      break


    hello_eve(line)
    ROOT.gEve.RemoveElement(ls)

