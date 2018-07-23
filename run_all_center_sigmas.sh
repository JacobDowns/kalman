for i in `seq 0 39`;
    do
            python run_center_sigmas.py $i >> filter/center_output/$i.txt &
    done
