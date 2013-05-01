#!/usr/bin/env ruby1.8
# Program to convert photonenergy to w

def e2w(n, m)
#  n = (n+0.938272)*(n+0.938272) - n*n
  n+m
end

puts e2w(ARGV[0].to_f, ARGV[1].to_f)
