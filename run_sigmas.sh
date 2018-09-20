for i in `seq 0 44`;
    do
            python run_sigmas.py $1 $i >> $1output/$i.txt &
    done
