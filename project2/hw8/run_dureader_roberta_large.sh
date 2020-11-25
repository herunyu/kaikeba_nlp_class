export PYTHONIOENCODING=utf-8
export DATA_DIR=data
export TASK_NAME=DuReader
export CUDA_VISIBLE_DEVICES=1
export OUTPUT_DIR=./roberta_large_output
# export MODEL_NAME_OR_PATH=$OUTPUT_DIR/$TASK_NAME/checkpoint-5000/
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext-large
python main.py --model_type roberta --model_name_or_path $MODEL_NAME_OR_PATH --do_train --data_dir=$DATA_DIR --max_seq_length 256 --per_gpu_train_batch_size 4 --do_lower_case --learning_rate 2e-5 --num_train_epochs 10.0 --output_dir $OUTPUT_DIR/$TASK_NAME/ --save_steps 1000 --overwrite_output_dir --max_answer_length 30
