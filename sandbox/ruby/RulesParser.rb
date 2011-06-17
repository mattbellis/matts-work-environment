################################################
##        ParameterRule class                 ##
################################################
class ParameterRule
  attr_accessor :terms

  ############################################
  def initialize
    @terms = 
      {
      "definition"=> "", 
      "free?"=> "", 
      "range"=> "", 
      "step"=> "", 
      "par"=> "", 
      "dpar"=> "" 
    }
  end

  ############################################
  def definitionSet(d)
    @terms["definition"]=d
  end
  ############################################
  def set(y)
    @terms["definition"]=y["definition"]
    @terms["free?"]=y["free?"]
    @terms["range"]=y["range"]
    @terms["step"]=y["step"]
    @terms["par"]=y["par"]
    @terms["dpar"]=y["dpar"]
  end

  ############################################
  def to_s
    puts "\t" + terms["definition"].to_s + " " + terms["free?"].to_s + " " \
      + terms["range"].to_s + " " + terms["step"].to_s + " " + terms["par"].to_s + " " + terms["dpar"].to_s
  end

  ############################################
  def dumpForInitialize
    puts "init_parameter('" + terms["definition"].to_s + "','value' => random_in" \
      + terms["range"].to_s + ",'step size' => " + terms["step"].to_s + ")"
  end
  ############################################
  def dumpForDefine
    if terms["free?"] == "true"
      puts "parameter('" + terms["definition"].to_s + "','free?' => " \
      + terms["free?"].to_s + ")"
    else
      puts "parameter('" + terms["definition"].to_s + "','free?' => " \
        + terms["free?"].to_s + ",'value' => " + terms["range"].to_s + ")"
    end
  end
  ############################################
  def dumpForList
    puts terms["free?"].to_s + "\t" + terms["range"].to_s + "\t" + terms["step"].to_s + "\t" + terms["definition"].to_s
  end

  ############################################
  def dumpForLikelihoodFunction
    puts "  wave.add_par('" + terms["definition"].to_s + "',par' => '" \
      + terms["par"].to_s   + "','dpar'=>'" + terms["dpar"].to_s + "')"
  end

end
################################################
################################################


################################################
##              AmpRule class                 ##
################################################
class AmpRule
  attr_accessor :type, :coherenceRule, :parRules

  ############################################
  def initialize
    @type = ""
    @coherenceRule = ""
    @parRules = [ ParameterRule.new ]
  end

  ############################################
  def typeSet(t)
    @type = t
  end

  ############################################
  def coherenceRuleSet(c)
    @coherenceRule = c
  end

  ############################################
  def parRuleAdd(r)
    offset = 0
    index = parRules.size - 1
    if parRules[index].terms["definition"] == ""
      offset = 0
    else
      offset = 1
    end
    parRules[index + offset] = ParameterRule.new
    parRules[index + offset].set(r)
  end

  ############################################
  def to_s
    parRules.each do |r|
      r.to_s
    end
    puts coherenceRule
  end

end
###############################################
################################################


################################################
##              Amp class                     ##
################################################
class Amp
  attr_accessor :name, :amp_rule, :matchesRule, :ampRe

  ############################################
  def initialize(name)
    @name = name
    @amp_rule = AmpRule.new
    @matchesRule = false
    @ampRe =  /([\w]+)=([\+\w.-]+)[:'\n']/
  end

  ############################################
  def dumpForLikelihoodFunction
    puts "  wave = new_wave('" + name.chomp + "')"
    amp_rule.parRules.each do |rp|
      rp.dumpForLikelihoodFunction
    end
    puts "  dataset('" + returnValue("dataset") + "').add wave"
  end

  ############################################
  def returnValue(tag)
    value = tag
    dumname = name.sub('.amps', '')
    dumname.scan(ampRe).each do |i|
      if i[0] == tag
        value = i[1] 
        break
      end
    end
    value
  end

  ############################################
  def matchRule(testRule)
    # Define the tag/value structure of the rule
    ruleRe = /([\w]+)=([\+\w.-]+)/
    @matchesRule = false
    # Loop over the tags in the rule
    testRule.type.scan(ruleRe).each do |j|
      foundOne = false
      # Loop over the tags in the amp and try to match up
      # both the tag and value
      dumname = name.sub('.amps', '')
      dumname.scan(ampRe).each do |i|
        if i[0] == j[0] && i[1].rindex(j[1]) != nil 
          foundOne = true
          break
        end
      end
      if foundOne
        @matchesRule = true
      else
        @matchesRule = false
        break
      end
    end
    if matchesRule
      setFromRule(testRule)
    end
  end

  ############################################
  ## To set the values from a rule
  def setFromRule(r)
    @amp_rule = AmpRule.new
    # Set the coherence
    dummy = ""
    r.coherenceRule.split('.').each do |x|
      foundOne = false
      dumname = name.sub('.amps', '')
      dumname.scan(ampRe).each do |i|
        if i[0] == x
          dummy += i[1] + "."
          foundOne = true
        end
      end
      if !foundOne
        dummy += x + "."
      end
    end
    @amp_rule.coherenceRuleSet(dummy.chop)

    # Set the pars
    count = 0
    while count < r.parRules.size 
      dummy = ""
      # Loop over the par rules terms
      r.parRules[count].terms["definition"].split('.').each do |x|
        foundOne = false
        # Loop over the amp name and grab what we will
        dumname = name.sub('.amps', '')
        dumname.scan(ampRe).each do |i|
          if i[0] == x
            dummy += i[1] + "."
            foundOne = true
          end
        end
        if !foundOne
          dummy += x + "."
        end
      end
      # Set the new names
      @amp_rule.parRuleAdd("definition"=>dummy.chop, 
                           "free?"=>r.parRules[count].terms["free?"], 
                           "range"=>r.parRules[count].terms["range"], 
                           "step"=>r.parRules[count].terms["step"], 
                           "par"=>r.parRules[count].terms["par"], 
                           "dpar"=>r.parRules[count].terms["dpar"] )
      count+=1
    end
  end

  ############################################
  def to_s
  end
end
###############################################
################################################

