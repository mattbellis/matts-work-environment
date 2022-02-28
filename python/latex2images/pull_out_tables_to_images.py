import sys
from TexSoup import TexSoup

import subprocess
import glob
import os


infile = open(sys.argv[1])
text = ""
for line in infile:
    text += line
print(text)

soup = TexSoup(text)

tables = list(soup.find_all('table'))
#tables = list(soup.find_all('figure'))
for count,t in enumerate(tables):
    output = "\\documentclass{article}\n"
    output += "\\usepackage{fullpage}\n"
    output += "\\setlength{\hoffset}{-0.5in}\n"

    output += "\\usepackage{tikz}\n"
    output += "\\usepackage[compat=1.1.0]{tikz-feynman}\n"

    output += "\\begin{document}\n"
    output += "\\pagestyle{empty}\n"
    output += str(t)
    output += "\n"
    output += "\\end{document}\n"
    print(output)

    outname = "file_image_{0}.png".format(count)
    if t.caption is not None:
        outname = (str(t.caption.string)[0:80]).replace(' ','_').replace('.','') + '.png'

    outfile = open("tmp.tex","w")
    outfile.write(output)
    outfile.close()

    subprocess.check_call(['pdflatex', 'tmp'])
    subprocess.check_call(['convert', '-density', '300', 'tmp.pdf', '-flatten', '-trim', '-quality', '100', outname])

    for cruft in glob.glob("tmp.*"):
          os.remove(cruft)


    

