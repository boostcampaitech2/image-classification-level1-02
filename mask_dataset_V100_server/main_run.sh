for seed in 1 7 77;
do
	for batch in 64 128;
	do
		for LR in 0.1 0.001 0.00001;
		do
			for WD in 0.001 0.0001 0.00001;
			do
				for p in 0.4 0.5 0.6;
				do
					for opt in Adam SGD;
					do
						for rrg in 0 5 15;
						do
                            echo seed_${seed}\|batch_${batch}\|LR_${LR}\|WD_${WD}\|opt_${opt}\|rrg_${rrg}
							python main.py \
								--SEED ${seed} \
								--BATCH_SIZE ${batch} \
								--LEARNING_RATE ${LR} \
								--WEIGHT_DECAY ${WD} \
								--Loos_2_portion ${p} \
								--OPTIMIZER ${opt} \
								--release_range ${rrg}
						done
					done
				done
			done
		done
	done
done

echo done
