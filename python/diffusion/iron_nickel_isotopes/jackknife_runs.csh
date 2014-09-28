@ num = -1

while ( $num > -29 )

    #python fit_delta.py FNDA2 Fe $num | grep 'Final value of beta'
    python fit_delta.py FNDA2 Ni $num | grep 'Final value of beta'

    @ num -= 1

end
