set infile = "github_names_and_accounts_UNIX.csv"

set work_dir = `pwd`

@ i = 1
while ($i < 35)

    set name = `sed "$i"p -n $infile | awk -F"," '{print $1'}`
    set account = `sed "$i"p -n $infile | awk -F"," '{print $2'}`

    set dir = `echo $name"_final_project"`

    echo $dir $name $account 

    #set url ="https://github.com/$account"
    set url = git@github.com:${account}\/CSIS\_200\_Final\_Project.git
    echo "url:"
    echo $url
    #git@github.com:Do03went/CSIS_200_Final_Project.git

    cd $dir
    pwd
    git clone $url
    cd $work_dir

    @ i += 1
end
