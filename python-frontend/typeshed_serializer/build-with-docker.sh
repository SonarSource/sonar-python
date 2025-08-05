#!/bin/sh

export USER_ID="$(id -u)"
export GROUP_ID="$(id -g)"
docker compose -f ./docker/docker-compose.yml --project-directory ./docker down
docker compose -f ./docker/docker-compose.yml --project-directory ./docker up --build
