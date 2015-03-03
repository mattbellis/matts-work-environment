#!/usr/bin/ruby1.8

class Address
  attr_accessor :street, :city, :state, :zip  
  def initialize
    @street = @city = @state = @zip = ""
  end
  def to_s
    "    " + @street + "\n" + \
    "    " + @city   + "\n" + \
    "    " + @state  + ", " + @zip  
  end
end

address = Address.new
address.street = "23 St George St."
address.city = "Pittsburgh"

puts
puts address.street + " " + address.city

puts address





