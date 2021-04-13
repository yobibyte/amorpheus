if [$1]; then
    # build for cpu, assuming we have gpu by default and go to the second branch
    cuda=cpu
else
    cuda=cu101 # dockerfile has it
fi
docker build --build-arg uid=$UID --build-arg user=$USER --build-arg cuda=$cuda -t amorpheus .
