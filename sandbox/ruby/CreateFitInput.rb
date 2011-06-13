#!/usr/bin/ruby1.8

require 'dumpOutput.rb'
require 'RulesParser.rb'
require 'rules.rb'

pars = []
lhTerm =          Hash[ String, Array[Amp] ]
lhTerms = Array [ Hash[ String, Array[Amp] ] ]
lhTerms.clear

File.open("listOfAmps.txt","r") do |f|
  f.each_line do |line| 
    amp = Amp.new(line)

    # $rule is global and included in rules.rb
    # Try and match up the amp with some rule
    $ampRule.each do |r|
      amp.matchRule(r)
    end

    # Fill the pars array
    amp.amp_rule.parRules.each do |rp|
      found = false
      pars.each do |par|
        if rp.terms["definition"] == par.terms["definition"]
          found = true
        end
      end
      if !found 
        pars.push(rp)
      end
    end

    # Fill the likelihood function
    found = false
    count = 0
    size = lhTerms.size 
    while count < size 
      if lhTerms[count]["coherence"] == amp.amp_rule.coherenceRule
        lhTerms[count]["amps"].push(amp)
        found = true
      end
      count+=1
    end
    if !found 
      lhTerm = {"coherence"=>amp.amp_rule.coherenceRule, "amps"=>Array[amp]}
      lhTerms.push(lhTerm)
    end
  end
end

# We've read in all the amps so
# dump the output.
dumpParameters(pars)
puts
dumpLikelihoodFunction(lhTerms)


