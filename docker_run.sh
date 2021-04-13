if [ $1 = "cpu" ]; then
    docker run -it -v `pwd`:/vitrin/amorpheus:rw --hostname $HOSTNAME --workdir /vitrin/amorpheus/modular-rl/src/scripts/ amorpheus
else
    docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -it -v `pwd`:/vitrin/amorpheus:rw --hostname $HOSTNAME --workdir /vitrin/amorpheus/modular-rl/src/scripts/ amorpheus
fi
