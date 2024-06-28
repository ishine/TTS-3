#!/bin/bash

# Docker version
#DOCKER_VERSION_TAG="2.2.0"

# Install path Espnet and Kaldi
INSTALL_PATH=/l/disk1/awstebas/


#Docker image version
#docker pull artifactory.cpqd.com.br/docker-dev/cpqd/tts/espnet:${DOCKER_VERSION_TAG}

nvidia-docker run -it -d --rm --ipc=host --network="host" \
  --name coqui_dev \
  -v /l/disk1/awstebas/coqui-ai:/workspace/coqui-ai\
  -v /l/disk1/awstebas/data:/l/disk1/awstebas/data \
  -v /l/disk1/awstebas/lhueda/data:/l/disk1/awstebas/lhueda/data \
  -v /l/disk1/awstebas/tts/templates:/workspace/templates \
  -v /l/disk1/awstebas/lhueda/github/unicamp-aiv2/TTS:/workspace/coqui-tts \
  -v /l/disk1/awstebas/tool_language/:/workspace/tool_language/ \
  -v /l/disk1/awstebas/lhueda/:/l/disk1/awstebas/lhueda \
  coqui_dev bash
