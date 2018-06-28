for i in `seq 0 39`;
    do
            python run_sigmas.py $i >> filter/prior1_output/$i.txt &
    done
