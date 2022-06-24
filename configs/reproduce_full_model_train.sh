export EXPNAME=reproduce_full_model

python train.py  \
          --dist-url 'tcp://localhost:10002' \
          --multiprocessing-distributed --world-size 1 --rank 0 \
          --model transunet_skip_res --residual_num 8  --norm in \
          --exp_name $EXP_NAME --training_mode all \
          --batch_size 16 --cos  --lr 1e-4 \
          --interpolated_normal \
          --netinput onlyiun_pol_vd \
          --dataset spwinter 
