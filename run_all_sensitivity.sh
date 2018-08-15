for i in `seq 0 39`;
    do
            python run_sensitivity.py $i >> sensitivity/output/$i.txt &
    done
