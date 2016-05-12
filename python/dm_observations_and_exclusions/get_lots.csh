@ i = 0

while ( $i < 1200 )

    set filename = "http://dmtools.brown.edu:8080/limits/"$i".xml"

    wget --user=bellis --password=lasong137 $filename

    @ i += 1

end
