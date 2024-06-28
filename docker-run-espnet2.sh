#!/bin/bash

# Docker version
#DOCKER_VERSION_TAG="2.2.0"
DOCKER_VERSION_TAG="2.4.0"  # dev

# Espnet install path
#INSTALL_PATH=/l/disk1/ic/N2/
INSTALL_PATH=/l/disk1/awstebas/

# Kaldi install path 
#KALDI_PATH=/l/disk1/ic/N2/
KALDI_PATH=/l/disk1/awstebas/

# Data path 
#DATA_PATH=/l/disk1/ic/N2/
DATA_PATH=/l/disk1/awstebas/

# Scripts-tts install path 
#SCRIPTS_TTS_PATH=/l/disk1/ic/N2/
SCRIPTS_TTS_PATH=/l/disk1/awstebas/

# Docker image version
#docker pull artifactory.cpqd.com.br/docker/cpqd/i2/tts/espnet:${DOCKER_VERSION_TAG}

nvidia-docker run -it -d --rm --ipc=host \
  --name espnet_dev \
  -v /l/disk1/awstebas/coqui-ai:/workspace/coqui-ai\
  -v /l/disk1/awstebas/data:/l/disk1/awstebas/data \
  -v /l/disk1/awstebas/lhueda/data:/l/disk1/awstebas/lhueda/data \
  -v /l/disk1/awstebas/tts/templates:/workspace/templates \
  -v /l/disk1/awstebas/lhueda/github/unicamp-aiv2/TTS:/workspace/coqui-tts \
  -v /l/disk1/awstebas/lhueda/:/l/disk1/awstebas/lhueda \
  artifactory.cpqd.com.br/docker/cpqd/i2/tts/espnet:${DOCKER_VERSION_TAG} bash
