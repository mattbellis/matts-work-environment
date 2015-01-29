#!/usr/bin/env ruby 


# Scan the surrounding network
system("sudo ifconfig eth2 down")
system("sudo ifconfig wlan0 up")
scan_output = `sudo iwlist wlan0 scan`

cell = Array.new
essid = Array.new
ap = Array.new
encryption = Array.new
strength = Array.new

scan_output.each_line{|line|

  if ( line.include?("Cell") )
    cell.push(line.split[1])
    ap.push(line.split[4])
  end

  if ( line.include?("ESSID") )
    essid.push(line.split(":")[1].strip)
  end

  if ( line.include?("Encryption") )
    encryption.push(line.split(":")[1].strip)
  end

  if ( line.include?("Quality") )
    strength.push(line.split("=")[1].split[0])
  end

}

max = cell.size()

if ( ARGV[0] == nil ) # Print out the choices
  for i in 0..max-1
    print "#{i}\t#{essid[i]}\t#{strength[i]}\t#{ap[i]}\t#{encryption[i]}\n"
  end
else # Or select a network to join
  choice = ARGV[0].to_i
  cmd = "sudo ifconfig wlan0 up"
  puts cmd
  system(cmd)
  cmd = "sudo iwconfig wlan0 essid #{essid[choice]} ap #{ap[choice]}"
  print "Attempting to join #{essid[choice]} at #{ap[choice]}\n"
  puts cmd
  system(cmd)
  system(cmd)
  cmd = "sudo dhclient wlan0"
  puts cmd
  system(cmd)
  #system("sudo dhclient wlan0")
end


