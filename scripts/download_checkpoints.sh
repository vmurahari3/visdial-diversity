#!/usr/bin/env bash

# SL checkpoints  
wget https://s3.amazonaws.com/visdial-diversity/checkpoints/2019_10_02/SL_QBOT.vd -O checkpoints-release/SL_QBOT.vd
wget https://s3.amazonaws.com/visdial-diversity/checkpoints/2019_10_02/SL_ABOT.vd -O checkpoints-release/SL_ABOT.vd

# SL Diverse Qbot checkpoint (referred to in the paper as Diverse-Q-Bot)
wget https://s3.amazonaws.com/visdial-diversity/checkpoints/2019_10_02/SL_DIV_QBOT.vd -O checkpoints-release/SL_DIV_QBOT.vd

# RL checkpoints ((Das et al (2017)))
wget https://s3.amazonaws.com/visdial-diversity/checkpoints/2019_10_02/ICCV_RL_QBOT.vd -O checkpoints-release/ICCV_RL_QBOT.vd
wget https://s3.amazonaws.com/visdial-diversity/checkpoints/2019_10_02/ICCV_RL_ABOT.vd -O checkpoints-release/ICCV_RL_ABOT.vd

#RL Checkpoints (RL Abot and RL Qbot finetuned from Diverse-Q-Bot)
wget https://s3.amazonaws.com/visdial-diversity/checkpoints/2019_10_02/RL_DIV_QBOT.vd -O checkpoints-release/RL_DIV_QBOT.vd
wget https://s3.amazonaws.com/visdial-diversity/checkpoints/2019_10_02/RL_DIV_ABOT.vd -O checkpoints-release/RL_DIV_ABOT.vd
