# Default values
DIM="256"
EXP_ID="point_push1_${DIM}"
TASK="OfflinePointPush1Gymnasium-v0"
TUNING_DIR="${EXP_ID}_finetune"
GPU_ID=3
CHECKPOINT=100
DDIM_SAMPLING_STEPS=200
TRAIN_BATCH_SIZE=32
GUIDANCE_WEIGHTS='{"w_obj": 1.0, "w_safe": 1.0}'

# Define the range of values for CHECKPOINT and GUIDANCE_SCALER
GUIDANCE_SCALER_VALUES=(10)
FINETUNE_STEPS_VALUES=(1)
FINETUNE_EPOCH_VALUES=(12)
FINETUNE_LR_VALUES=(1e-5)
ALPHA_VALUES=(0.9)


for ALPHA in "${ALPHA_VALUES[@]}"; do
    for GUIDANCE_SCALER in "${GUIDANCE_SCALER_VALUES[@]}"; do
        for FINETUNE_EPOCH in "${FINETUNE_EPOCH_VALUES[@]}"; do
            for FINETUNE_STEPS in "${FINETUNE_STEPS_VALUES[@]}"; do
                for FINETUNE_LR in "${FINETUNE_LR_VALUES[@]}"; do
                    TUNING_ID="${FINETUNE_LR}-${GUIDANCE_SCALER}-${ALPHA}-${FINETUNE_STEPS}-${FINETUNE_EPOCH}"

                    # Print configuration
                    echo "Running inference with:"
                    echo "Experiment ID: $EXP_ID"
                    echo "GPU ID: $GPU_ID"
                    echo "Checkpoint: $CHECKPOINT"
                    echo "Guidance scaler: $GUIDANCE_SCALER"
                    echo "Guidance weights: $GUIDANCE_WEIGHTS"
                    echo "Loss weights: $LOSS_WEIGHTS"
                    echo "DDIM sampling steps: $DDIM_SAMPLING_STEPS"

                    python run_inference.py \
                        --task "$TASK" \
                        --gpu_id "$GPU_ID" \
                        --exp_id "$EXP_ID" \
                        --tuning_dir "$TUNING_DIR" \
                        --tuning_id "$TUNING_ID" \
                        --dim "$DIM" \
                        --checkpoint "$CHECKPOINT" \
                        --finetune_epoch "$FINETUNE_EPOCH" \
                        --finetune_steps "$FINETUNE_STEPS" \
                        --finetune_lr "$FINETUNE_LR" \
                        --guidance_weights "$GUIDANCE_WEIGHTS" \
                        --loss_weights '{"loss_train": 1.0, "loss_test": 0.0}' \
                        --guidance_scaler "$GUIDANCE_SCALER" \
                        --train_batch_size "$TRAIN_BATCH_SIZE" \
                        --ddim_sampling_steps "$DDIM_SAMPLING_STEPS" \
                        --alpha "$ALPHA"
                    done
                done
            done
        done
    done
done