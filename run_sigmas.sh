for i in `seq 0 46`;
    do
            python run_sigmas.py $1 $i >> $1output/$i.txt &
    done
