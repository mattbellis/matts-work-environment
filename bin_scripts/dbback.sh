mysqldump --tab=junkDBbackup --user=root --password=opiate77 --opt wikidb
mysqlhotcopy -p opiate77 wikidb junkDBbackup/
