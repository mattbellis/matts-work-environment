#set infile = "github_names_and_accounts_UNIX.csv"
set infile = "usernames.csv"

set work_dir = `pwd`

@ i = 1
while ($i < 35)

    set name = `sed "$i"p -n $infile | awk -F"," '{print $1'}`
    set account = `sed "$i"p -n $infile | awk -F"," '{print $2'}`

    echo "THRER: " $account

    set dir = `echo $account`

    echo $dir $name $account 

    #set url ="https://github.com/$account"
    set url = "git@github.com:$account/CSIS200_Final_Project_F16.git"
    echo "url:"
    echo $url
    #git@github.com:Do03went/CSIS_200_Final_Project.git

    echo $dir
    mkdir -p $dir
    cd $dir
    pwd
    echo git clone $url
         git clone $url
    cd $work_dir

    @ i += 1
end
