# docker start -ai verl_test

docker run -it \
  --name verl_test \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  --shm-size=64g \
  -v /data:/data \
  -v /var/run/docker.sock:/var/run/docker.sock \
  verl:vllm024.dev2-uniagent-runable \
  bash