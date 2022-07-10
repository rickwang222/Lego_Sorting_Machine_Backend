#!/bin/bash

PATH="/home/pi/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games"

trap 'kill 0' SIGINT; 
ssh -tt -L 9001:localhost:9001 mart4322@kia09.ece.umn.edu ./start_lego_proj_inf.sh &
cd /home/pi/UIServer/
python /home/pi/UIServer/server.py &
while :
do
	sleep 1
done
