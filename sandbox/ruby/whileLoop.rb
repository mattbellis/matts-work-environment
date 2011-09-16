#!/usr/bin/ruby1.8
#
print "enter? "
while entry = gets.chop
  break if entry.empty?
  puts entry
  print "enter? "
end
