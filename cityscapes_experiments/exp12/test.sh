for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
    echo "Testing for model $i \n"
    echo "Testing for model $i \n" >> testing_progress.txt

    python -u driver.py cuda:6 $i True False ../../data_configs/cityscapes.yml 2 'saved_models4_dice/client_super_best_val.pth' >> testing_progress2.txt
done