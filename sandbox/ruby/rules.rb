require 'RulesParser.rb'


######################
$ampRule = [ AmpRule.new, AmpRule.new, AmpRule.new, AmpRule.new  ]

# Treat the magnetic multipoles differently
$ampRule[0].typeSet("channel=s:mp=M")
$ampRule[0].coherenceRuleSet("mz_g.mz_i.mz_f.dataset.final_state")

$ampRule[0].parRuleAdd("definition"=>"gP.mod.channel.isospin.jp",
                        "free?"=>"true",
                        "range"=>"(0,6.28)", 
                        "step"=>"0.01", 
                        "par"=>"cos(x)",  
                        "dpar"=>"-sin(x)") 

$ampRule[0].parRuleAdd("definition"=>"gP.phase.channel.isospin.jp",
                       "free?"=>"true",
                       "range"=>"(0,6.28)",
                       "step"=>"0.01",
                       "par"=>"exp(i*x)",
                       "dpar"=>"i*exp(i*x)")

$ampRule[0].parRuleAdd("definition"=>"gD.mod.channel.isospin.jp.final_state.L.S",
                       "free?"=>"true",
                       "range"=>"(-10,10)",
                       "step"=>"0.01",
                       "par"=>"x",       
                       "dpar"=>"1")


# Treat the electric multipoles differently
$ampRule[1].typeSet("channel=s:mp=E")
$ampRule[1].coherenceRuleSet("mz_g.mz_i.mz_f.dataset.final_state")

$ampRule[1].parRuleAdd("definition"=>"gP.mod.channel.isospin.jp",
                        "free?"=>"true",
                        "range"=>"(0,6.28)", 
                        "step"=>"0.01", 
                        "par"=>"sin(x)",  
                        "dpar"=>"cos(x)") 

$ampRule[1].parRuleAdd("definition"=>"gP.phase.channel.isospin.jp",
                       "free?"=>"true",
                       "range"=>"(0,6.28)",
                       "step"=>"0.01",
                       "par"=>"exp(i*x)",
                       "dpar"=>"i*exp(i*x)")

$ampRule[1].parRuleAdd("definition"=>"gD.mod.channel.isospin.jp.final_state.L.S",
                       "free?"=>"true",
                       "range"=>"(-10,10)",
                       "step"=>"0.01",
                       "par"=>"x",       
                       "dpar"=>"1")


# This is for s-channel amplitudes with J=1/2 and only one coupling
$ampRule[2].typeSet("channel=s:jp=1")
$ampRule[2].coherenceRuleSet("mz_g.mz_i.mz_f.dataset.final_state")

$ampRule[2].parRuleAdd("definition"=>"gP.mod.channel.isospin.jp",
                        "free?"=>"true",
                        "range"=>"(-10,10)", 
                        "step"=>"0.01", 
                        "par"=>"x",  
                        "dpar"=>"1") 

$ampRule[2].parRuleAdd("definition"=>"gP.phase.channel.isospin.jp",
                       "free?"=>"false",
                       "range"=>"0.0",
                       "step"=>"0.0",
                       "par"=>"exp(i*x)",
                       "dpar"=>"i*exp(i*x)")


# This is for t-channel amplitudes 
$ampRule[3].typeSet("channel=t")
$ampRule[3].coherenceRuleSet("mz_g.mz_i.mz_f.dataset.final_state")

$ampRule[3].parRuleAdd("definition"=>"gP.mod.channel.jp.lorentz",
                        "free?"=>"true",
                        "range"=>"(-10,10)", 
                        "step"=>"0.01", 
                        "par"=>"x",  
                        "dpar"=>"1") 





