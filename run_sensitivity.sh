for i in `seq 0 44`;
    do
	python run_sensitivity.py $1 $2 $i >> transform_long/$1_$2/output/$i.txt &
    done
