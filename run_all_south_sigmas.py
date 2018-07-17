for i in `seq 0 39`;
    do
            python run_south_sigmas.py $i >> filter/south_output/$i.txt &
    done
