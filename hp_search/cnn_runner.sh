count=0
for LR in 1 0.1 0.01 0.001 ;
do
    for OPT in Adam SGD ;
    do
        count=$((${count}+1))
        python cnn.py \
            --EPOCH 1 \
            --OPTIMIZER ${OPT} \
            --LEARNING_RATE ${LR} \
            --EXP_NUM ${count}
    done
done
