#!/bin/bash

NODE_TYPE=$1
CONFIG_FILE=${2:-"/mnt/nfs/naifu/config/train_noob.yaml"}

# 设置节点参数
if [ "$NODE_TYPE" = "node0" ]; then
    NODE_RANK=0
elif [ "$NODE_TYPE" = "node1" ]; then
    NODE_RANK=1
elif [ "$NODE_TYPE" = "node2" ]; then
    NODE_RANK=2
else
    echo "错误: 无效节点类型"
    exit 1
fi

# 使用torchrun启动（更可靠）
torchrun \
    --nproc_per_node=8 \
    --nnodes=3 \
    --node_rank=$NODE_RANK \
    --master_addr=192.168.16.3 \
    --master_port=29500 \
    trainer.py --config $CONFIG_FILE