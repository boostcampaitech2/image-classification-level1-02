count=1
while [ ${count} -le 10 ];
do
	python main.py --RANDOM_SEARCH True --SPLIT True --EXP_NUM ${count}
	count=$((${count}+1))
	echo "EXP_NUM ${count} is done."
done
