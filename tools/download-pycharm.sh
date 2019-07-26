#!/bin/sh

set -eu

PYCHARM_VERSION=2019.1.3

curl -L -O https://download-cf.jetbrains.com/python/pycharm-community-${PYCHARM_VERSION}.tar.gz
tar xzf pycharm-community-${PYCHARM_VERSION}.tar.gz
rm pycharm-community-${PYCHARM_VERSION}.tar.gz

cd pycharm-community-${PYCHARM_VERSION}/lib
mvn install:install-file -Dfile=extensions.jar -DgroupId=com.jetbrains.pycharm -DartifactId=extensions -Dversion=${PYCHARM_VERSION} -Dpackaging=jar
mvn install:install-file -Dfile=openapi.jar -DgroupId=com.jetbrains.pycharm -DartifactId=openapi -Dversion=${PYCHARM_VERSION} -Dpackaging=jar
mvn install:install-file -Dfile=platform-api.jar -DgroupId=com.jetbrains.pycharm -DartifactId=platform-api -Dversion=${PYCHARM_VERSION} -Dpackaging=jar
mvn install:install-file -Dfile=platform-impl.jar -DgroupId=com.jetbrains.pycharm -DartifactId=platform-impl -Dversion=${PYCHARM_VERSION} -Dpackaging=jar
mvn install:install-file -Dfile=pycharm.jar -DgroupId=com.jetbrains.pycharm -DartifactId=pycharm -Dversion=${PYCHARM_VERSION} -Dpackaging=jar
mvn install:install-file -Dfile=pycharm-pydev.jar -DgroupId=com.jetbrains.pycharm -DartifactId=pycharm-pydev -Dversion=${PYCHARM_VERSION} -Dpackaging=jar
mvn install:install-file -Dfile=resources_en.jar -DgroupId=com.jetbrains.pycharm -DartifactId=resources_en   -Dversion=${PYCHARM_VERSION} -Dpackaging=jar
mvn install:install-file -Dfile=util.jar -DgroupId=com.jetbrains.pycharm -DartifactId=util -Dversion=${PYCHARM_VERSION} -Dpackaging=jar
mvn install:install-file -Dfile=jps-model.jar -DgroupId=com.jetbrains.pycharm -DartifactId=jps-model -Dversion=${PYCHARM_VERSION} -Dpackaging=jar
