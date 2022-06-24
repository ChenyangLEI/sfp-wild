export EXPNAME=0729_unet_deepsfp_onlyiun
# python eval_sfpwild.py --pred_root ../results/$EXPNAME/sfpwild_combined --test_txt ../data/iccv2021/sfpwild_test_intra.txt
# python eval_sfpwild.py --pred_root ../results/$EXPNAME/sfpwild_combined --test_txt ../data/iccv2021/sfpwild_test_printed.txt
# python eval_sfpwild.py --pred_root ../results/$EXPNAME/sfpwild_combined --test_txt ../data/iccv2021/sfpwild_combined.txt

python eval.py --pred_root ../results/$EXPNAME/sfpwild_test_inter_combined --test_txt ../data/iccv2021/sfpwild_test_inter.txt
python eval.py --pred_root ../results/$EXPNAME/sfpwild_test_inter_combined --test_txt ../data/iccv2021/sfpwild_test_printed.txt
python eval.py --pred_root ../results/$EXPNAME/sfpwild_test_inter_combined --test_txt ../data/iccv2021/sfpwild_test_inter_combined.txt


python eval.py --pred_root ../results/$EXPNAME/deepsfp --test_txt deepsfp.txt
