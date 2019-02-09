for i in `seq 0 45`;
    do
	python2 run_sigmas_sensitivity.py $1 $i >> $1/output/$i.txt &
	sleep 1
    done
