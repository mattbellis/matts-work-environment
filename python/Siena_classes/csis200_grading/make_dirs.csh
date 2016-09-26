set infile = "github_names_and_accounts.csv"

foreach name (`cat $infile | awk -F"," '{print $1'}`)
    echo $name 
    mkdir $name"_final_project"
end
