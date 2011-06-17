#!/bin/tcsh 

foreach ntp ( 1 2 )
    foreach p_or_e ( "--pure" "--embed" )
        foreach fixed_or_not ( "--fixed-num" " " )
            foreach num(0 10 20 30 40 50 60 70 80 90 100    2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36)
                python run_a_set_of_fits.py --my-python $MY_PYTHON \
                                            --baryon LambdaC \
                                            --ntp $ntp \
                                            --pass 0 \
                                            --step 5 \
                                            --num-sig $num \
                                            $p_or_e \
                                            --sideband-first \
                                            --num-fits 1000 \
                                            $fixed_or_not \
                                            --batch
            end
        end
    end
end


