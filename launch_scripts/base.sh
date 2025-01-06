# 1) finetune on all datasets
# 2) eval single task
# 3) task addition

python /kaggle/working/polito-task-arithmetic/finetune.py \
--data-location=/kaggle/working/datasets/ \
--save=/kaggle/working/checkpoints/ \
--batch-size=32 \
--lr=1e-4 \
--wd=0.0


#python eval_single_task.py \
#--data-location=/content/drive/MyDrive/Task_Arithmetic_Datasets/datasets/ \
#--save=/content/drive/MyDrive/Task_Arithmetic_Checkpoints/

#python eval_task_addition.py \
#--data-location=/content/drive/MyDrive/Task_Arithmetic_Datasets/datasets/ \
#--save=/content/drive/MyDrive/Task_Arithmetic_Checkpoints/
