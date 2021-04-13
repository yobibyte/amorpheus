cd ../..
for seed in 1 2 3;
do
	sleep 5
        python3 main.py \
          --morphologies nervenet \
          --seed $seed \
          --td \
          --bu \
          --lr 0.0001 \
          --label nervenet_walkers_smp&
done
cd scripts

