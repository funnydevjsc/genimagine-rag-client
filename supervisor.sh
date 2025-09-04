#!/bin/sh
cd /home/funnydev/genimagine-bot
echo > supervisor.log

/usr/bin/conda init bash
. /home/funnydev/anaconda3/etc/profile.d/conda.sh
/usr/bin/conda activate genimagine_bot

lsof -ti tcp:8000 | xargs -r kill -9

cd /home/funnydev/genimagine-bot/docker_qdrant
docker compose down
rm -f .lock
docker compose up -d

cd /home/funnydev/genimagine-bot
export OLLAMA_NUM_GPU_LAYERS=32
/home/funnydev/anaconda3/envs/genimagine_bot/bin/python main.py
