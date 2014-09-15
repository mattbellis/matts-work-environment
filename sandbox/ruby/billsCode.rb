#!/usr/bin/ruby1.8

File.open("xxx.txt","r") do |f|
  f.each_line do |line| 
    #print "#{line}"
    if (line.match(/ (.+)/))
      match, newline = *(line.match(/ (.+)/))
      print "#{newline}\n"
    elsif
      print "#{line}"
    end
  end
end
