#!/usr/bin/ruby1.8

print "hi\n"

topology = "pippim_p"

if ( topology == nil )
  print "No topology given!!!!!"
  exit(-1)
else
  tnum = 2
  print "topology: #{topology}\n"
  if ( topology == "ppip_pim" )
    print "a"
    tnum = 1 
  elsif ( topology == "ppim_pip" )
    print "b"
    tnum = 2 
  elsif ( topology == "pippim_p" )
    print "c\n"
    tnum = 3 
  end
  print "topology: #{topology}\n"
  print "tnum: #{tnum}\n"
end



#print "#{b[0]}\n"

