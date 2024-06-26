#!/bin/sh

export USER_ID="$(id -u)"
export GROUP_ID="$(id -g)"
#echo $USER_ID
#echo $GROUP_ID
docker compose -f ./docker/docker-compose.yml --project-directory ./docker up