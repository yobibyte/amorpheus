cd ../..
for seed in 1 2 3;
do
	sleep 5
        python3 main.py \
          --custom_xml environments/cheetahs \
          --seed $seed \
          --td \
          --bu \
          --lr 0.0001 \
          --label cheetah_smp&
done
cd scripts

