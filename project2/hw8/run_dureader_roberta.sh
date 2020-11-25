export PYTHONIOENCODING=utf-8
export DATA_DIR=data
export TASK_NAME=DuReader
export CUDA_VISIBLE_DEVICES=1
python main.py --model_type roberta --model_name_or_path hfl/chinese-roberta-wwm-ext --do_train --data_dir=$DATA_DIR --max_seq_length 256 --per_gpu_train_batch_size 16 --do_lower_case --learning_rate 2e-5 --num_train_epochs 10.0 --output_dir ./roberta_output/$TASK_NAME/ --save_steps 500 --overwrite_output_dir #--overwrite_cache 
