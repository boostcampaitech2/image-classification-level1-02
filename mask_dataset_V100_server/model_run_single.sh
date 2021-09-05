seed=1
batch=64
LR=0.00001
WD=0.00011
p=0.6
opt=Adam
rrg=0
echo seed_${seed}\|batch_${batch}\|LR_${LR}\|WD_${WD}\|opt_${opt}\|rrg_${rrg}
python main.py \
    --SEED ${seed} \
    --BATCH_SIZE ${batch} \
    --LEARNING_RATE ${LR} \
    --WEIGHT_DECAY ${WD} \
    --Loos_2_portion ${p} \
    --OPTIMIZER ${opt} \
    --release_range ${rrg} \
    --TOTAL_EPOCH 100
echo done
