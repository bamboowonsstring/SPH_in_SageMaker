#!/usr/bin/env bash

# リポジトリの管理
git config --global user.name bamboowonsstring
git config --global user.email extra.excramattion1@gmail.com
#Pfpy update
if [ "$1" == "" ]
then
 git -C ./ParticleFluidPy pull https://github.com/class-snct-rikitakelab/ParticleFluidPy master
 git add ParticleFluidPy
 git commit -m "update submodule"
fi

# pull
if [ "$1" == "pull" ]
then
 git pull
 git submodule update
 exit 0
fi

#commit
if [ "$1" == "commit" ]
then
 git add .
 git commit -m "update"
fi