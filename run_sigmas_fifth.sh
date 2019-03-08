seq 5000 | parallel -j45 "python run_sigma.py $1 {} |rotatelogs -n 1 $1output/{%}.txt 1M"
