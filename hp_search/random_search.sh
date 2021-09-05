count=0
while [ ${count} -le 10 ];
do
count=$((${count}+1))
python cnn.py --RandomMode True --EXP_NUM ${count}
done