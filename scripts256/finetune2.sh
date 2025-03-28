# Default values
DIM="256"
EXP_ID="point_goal1_${DIM}"
TASK="OfflinePointGoal1Gymnasium-v0"
TUNING_DIR="finetune"
POST_TRAIN_ID="posttrain/1e-5-10-0.9-1"
GPU_ID=2
CHECKPOINT=0
DDIM_SAMPLING_STEPS=200
FINETUNE_STEPS=1
FINETUNE_EPOCH=12

# Define the range of values for CHECKPOINT and GUIDANCE_SCALER
GUIDANCE_SCALER_VALUES=(120)
FINETUNE_LR_VALUES=(1e-5)
GUIDANCE_WEIGHTS_VALUES=('{"w_obj": 1.0, "w_safe": 1.0}')
ALPHA_VALUES=(0.9)

# Grid search over CHECKPOINT and GUIDANCE_SCALER
for ALPHA in "${ALPHA_VALUES[@]}"; do
    for GUIDANCE_SCALER in "${GUIDANCE_SCALER_VALUES[@]}"; do
        for FINETUNE_LR in "${FINETUNE_LR_VALUES[@]}"; do
            for GUIDANCE_WEIGHTS in "${GUIDANCE_WEIGHTS_VALUES[@]}"; do
                TUNING_ID="Q0-${ALPHA}-${GUIDANCE_WEIGHTS}-${FINETUNE_LR}-${GUIDANCE_SCALER}"

                # Print configuration
                echo "Running inference with:"
                echo "Experiment ID: $EXP_ID"
                echo "GPU ID: $GPU_ID"
                echo "Checkpoint: $CHECKPOINT"
                echo "Guidance scaler: $GUIDANCE_SCALER"
                echo "Guidance weights: $GUIDANCE_WEIGHTS"
                echo "Loss weights: $LOSS_WEIGHTS"
                echo "DDIM sampling steps: $DDIM_SAMPLING_STEPS"
                echo "DDIM sampling eta: $DDIM_ETA"

                python run_inference.py \
                    --finetune_set "test" \
                    --task "$TASK" \
                    --use_guidance \
                    --backward_finetune \
                    --gpu_id "$GPU_ID" \
                    --exp_id "$EXP_ID" \
                    --tuning_dir "$TUNING_DIR" \
                    --tuning_id "$TUNING_ID" \
                    --post_train_id "$POST_TRAIN_ID" \
                    --dim "$DIM" \
                    --checkpoint "$CHECKPOINT" \
                    --finetune_epoch "$FINETUNE_EPOCH" \
                    --finetune_steps "$FINETUNE_STEPS" \
                    --finetune_lr "$FINETUNE_LR" \
                    --guidance_weights "$GUIDANCE_WEIGHTS" \
                    --loss_weights "$LOSS_WEIGHTS" \
                    --guidance_scaler "$GUIDANCE_SCALER" \
                    --ddim_sampling_steps "$DDIM_SAMPLING_STEPS" \
                    --alpha "$ALPHA"
            done
        done
    done
done