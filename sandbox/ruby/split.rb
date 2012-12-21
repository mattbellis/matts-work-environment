#!/usr/bin/ruby1.8

a = "z=1:n=2:m=4"

b = Hash.new

a.split(":").each{|f| 
  tag = f.split("=")[0]
  val = f.split("=")[1]
  #print "#{tag}\t#{val}\n"
  b[tag] = val 
  print "#{tag}\t#{b[tag]}\n"
}

#print "#{b[0]}\n"

