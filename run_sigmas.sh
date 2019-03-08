for i in `seq 0 45`;
    do
        python run_sigmas.py $1 $i | rotatelogs -n 5 $1output/$i.txt 1M &
	sleep .1
    done
