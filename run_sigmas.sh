for i in `seq 0 39`;
    do
            python run_sigmas.py $1 $i >> $1output/$i.txt &
    done
