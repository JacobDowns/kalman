for i in `seq 0 39`;
    do
            python run_north_sigmas.py $i >> filter/north_output/$i.txt &
    done
