#!/usr/bin/ruby
require 'net/http'

host = "www.smartmoney.com"
link = "/eqsnaps/index.cfm?story=snapshot&symbol="+ARGV[0]

begin

  # Create a new HTTP connection
  httpCon = Net::HTTP.new( host, 80 )

  # Perform a HEAD request
  resp = httpCon.get( link, nil )

  stroffset = resp.body =~ /class="price">/

  subset = resp.body.slice(stroffset+14, 10)

  limit = subset.index('<')

  print ARGV[0] + " current stock price " + subset[0..limit-1] +
                          " (from stockmoney.com)\n"

end

