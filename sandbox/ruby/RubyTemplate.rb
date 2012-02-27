#!/usr/bin/env ruby

# Include any necessary extra ruby files
require 'optparse'

verbose = 0

printf "Hi\n"

#
options = OptionParser.new do |opts|
  #
  opts.on("-o", "--option [ARG]", "Option description") do |opt|
    app['option'] = opt
  end
  #
  opts.on("-v", "--verbose [ARG]", "Set the level of verbosity. Can call this multiple times.") do |opt|
    verbose += 1
  end
  #
end
#

#
begin
  #
  options.parse!(ARGV)
  #
rescue OptionParser::ParseError => e
  #
  puts e
  #
end

printf "Verbosity: #{verbose}\n"
