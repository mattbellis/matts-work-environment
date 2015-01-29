#######################
# Dump the output for the parmater list
#######################

def dumpParameters(pars)
  puts 
  puts '#-----------------------------------------------------------------------------#'
  puts '#                        Define All MINUIT Parameters                         #'
  puts '#-----------------------------------------------------------------------------#'
  puts 'define_parameters { # only executed once during pre-fit set up '
  pars.each do |par|
    par.dumpForDefine
  end
  puts '}'

  puts 
  puts '#-----------------------------------------------------------------------------#'
  puts '#             Initialize MINUIT Parameters for Next Fit Iteration             #'
  puts '#-----------------------------------------------------------------------------#'
  puts 'initialize_parameters { # executed before each iteration'
  pars.each do |par|
    if par.terms["free?"] == "true"
      par.dumpForInitialize
    end
  end
  puts '}'

end



#######################
# Dump the output for the likelihood function
#######################

def dumpLikelihoodFunction(lhTerms)

  puts '#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#'
  puts '#-----------------------------------------------------------------------------#'
  puts '#                              Dataset \'' + lhTerms[0]["amps"][0].returnValue("dataset") + \
     '\'                                  #'
     puts '#-----------------------------------------------------------------------------#'
     puts 'define_dataset { # only executes during initial pre-fit set up'
     lhTerms.each do |lt|
    puts '  #'
    puts '  # Incoherent set of waves'
    puts '  #'
    puts '  dataset(\'' + lt["amps"][0].returnValue("dataset") + '\').new_incoherent_waveset'
    puts '  #'
    lt["amps"].each do |a|
      a.dumpForLikelihoodFunction
      puts '  #'
    end
     end
  puts '}'
  puts '#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#'

end


