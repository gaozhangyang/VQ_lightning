
export PYTHONPATH="/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/BSQ_lightning:$PYTHONPATH"

torchrun --nnodes=1 --nproc_per_node=8  task1_VQ/main.py --config_name VQ-4096 --offline 1 --ex_name 'VQ-4096-vvq' --vq_type 'vvq' --gpus_per_node 8 --num_nodes 1 --batch_size 64

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python task1_VQ/main.py --config_name VQ-4096 --offline 0 --ex_name 'VQ-4096-vvq-lr1e-5' --vq_type 'vvq' --num_nodes 1 --batch_size 64

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python task1_VQ/main.py --config_name RotFSQ-4096 --offline 0 --ex_name 'VQ-4096-rotfsq-lr1e-4' --vq_type 'rotfsq' --num_nodes 1 --batch_size 64

CUDA_VISIBLE_DEVICES='0,3,5,6,7'

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python task1_VQ/main.py --config_name VQ-4096 --offline 0 --ex_name 'VQ-4096-lr1e-4-clsloss' --vq_type 'vvq' --num_nodes 1 --batch_size 64

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python task1_VQ/main.py --config_name VQ-4096 --offline 0 --ex_name 'VQ-4096-lr1e-4-clsloss-alpha0.1' --vq_type 'vvq' --num_nodes 1 --batch_size 64

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python task1_VQ/main.py --config_name VQ-4096 --offline 0 --ex_name 'VQ-4096-lr1e-4-clsloss-alpha0.01' --vq_type 'vvq' --num_nodes 1 --batch_size 64