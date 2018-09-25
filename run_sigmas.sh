for i in `seq 0 45`;
    do
            python run_sigmas.py $1 $i >> $1output/$i.txt &
    done
