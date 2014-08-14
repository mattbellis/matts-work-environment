#!/usr/bin/ruby
#
libdir = `root-config --libdir`
rootlib = libdir.strip + "/libRuby"
require "#{rootlib}"


#graphs = Hash[ String, Array[TGraphErrors] ]

hash = Hash.new

gr = Array.new
for i in 0...10
  gr[i] = TGraphErrors.new
  name = i.to_s + "number"
  value = (-2)**i.to_i
  print "value: #{value}\n"
  hash[i] = {'name'=>name, 'value'=>value.to_f, 'graph'=>gr[i], 'is_master'=>false}
  hash[i]['graph'].SetName("thisname")
end

#print "\nval: #{hash[5]['value']}\n"
#print "\n0: #{hash[5].key['value']}\n"

#print "#{hash.sort{|x,y| y<=>x}}\n"
#print "#{hash.sort{|a,b| a[0]<=>b[0]}}\n"


for i in 0...hash.size

  print "5numbercheck: #{hash[i].has_value?('5number')}\n"
  print "key name check: #{hash[i].has_key?('name')}\n"
  print "#{i} #{hash[i]['value']}\n"
  print "#{i} #{hash[i]['name']}\n"

end

