
#echo $0
#echo $1

#echo $(($2 - 1))
for i in $(seq 0 2)
do
    python run_sensitivity_steady.py $1 $i >> sensitivity/$1/out_$i.txt &
done
