using HDF5
using Plots

fname = "output.h5"

fid = h5open(fname, "r")

names = keys(fid)

for name in names
    println(name)
end

jet = read(fid,"jet")

jet_fields = keys(jet)
for field in jet_fields
  println(field)
end

e = jet["e"]

h = histogram(e,bins=25)

savefig("julia_plot_output.png")


gui()



