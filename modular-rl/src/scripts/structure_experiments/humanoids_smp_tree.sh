cd ../..
for seed in 1 2 3;
do
	sleep 5
        python3 main.py \
          --custom_xml environments/humanoids \
          --seed $seed \
          --td \
          --bu \
          --lr 0.0001 \
          --observation_graph_type tree \
          --label humanoids_smp_tree&
done
cd scripts

