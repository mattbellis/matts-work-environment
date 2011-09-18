#!/usr/bin/env ruby

require "ftools"  # for File.copy

IO.foreach("smearing_values.asc") { |params|
  next if params =~ /^#/
  numbers = params.scan /[-+]?\d*\.?\d+/
  if numbers[0].to_f < 10 then
    dirname = "job0#{numbers[0]}"
  else
    dirname = "job#{numbers[0]}"
  end

  if File.exist?(dirname) then
    puts "Skipping #{dirname} because it exists"
  else
    puts "Creating #{dirname}"
    Dir.mkdir(dirname)
    outFile = File.new("#{dirname}/run_qqq.pl", "w")
    IO.foreach("run_qqq.pl-template") { |line|
      line.sub!("__GAUSS_RAD__", numbers[1])
      line.sub!("__GAUSS_ITR__", numbers[2])
      line.sub!("__STOUT_RHO__", numbers[3])
      line.sub!("__STOUT_ITR__", numbers[4])
      line.sub!("__SMEARING_PARAMS__", "G" + numbers[1].tr('.', 'p') +
                                       "_" + numbers[2] +
                                      "_S" + numbers[3].tr('.', 'p') +
                                       "_" + numbers[4])
      outFile << line
    }
    outFile.chmod(0755)
    outFile.close 

    filenames = [ "ib.csh", "wl_5p5_x2p38_um0p4125.list" ]
    filenames.each { |filename| File.copy(filename, "#{dirname}/#{filename}") }
  end
}
