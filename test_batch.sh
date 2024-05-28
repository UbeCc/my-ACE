num_gpus=8
iter=0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for lr in 1e-3; do
        for per_alpha in 0.7; do
            for per_beta in 0.6; do
                for per_eps in 0.1; do
                    for tau in 0.1; do
                        for gamma in 0.99; do
                            for env_name in Walker2d-v4; do
                                for agent in td3; do
                                    echo agent is ${agent}
                                    device=$((iter % num_gpus))
                                    ((iter++))
                                    python main.py              \
                                        device=${device} \
                                        +agent=${agent} \
                                        agent.lr_actor=${lr}          \
                                        agent.lr_critic=${lr}         \
                                        buffer.use_per=True     \
                                        buffer.nstep=1   \
                                        buffer.per_alpha=${per_alpha}  \
                                        buffer.per_beta=${per_beta}    \
                                        buffer.per_eps=${per_eps}      \
                                        agent.tau=${tau}              \
                                        agent.gamma=${gamma}          \
                                        env_name=${env_name}  \
                                    &
                                done        
                            done
                        done
                    done
                done
            done
        done
done