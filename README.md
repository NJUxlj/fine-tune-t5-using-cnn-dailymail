# Fine-Tuning google-T5 on CNN-DailyMail, using huggingface trainer, scripts, peft, trl, as a standard SOP
我们在cnn-dailymail数据集上微调google-T5模型，微调方法包括：使用scripts、trainer、pytorch、peft、trl。另外，我们设计一个数据集转换函数，将bbc_news数据集转为了cnn_dailymail的格式。另外，还包含了多GPU微调，上传huggingface_hub的一整套标准SOP





## 使用脚本微调

![image](https://github.com/user-attachments/assets/fc23ff70-a642-4fb2-994a-427780de3306)
![image](https://github.com/user-attachments/assets/eba10696-984f-4a40-aa61-6947b482ff76)


### 使用Torchrun来分布式微调

```shell
# 这是用于多GPU分布式训练的PyTorch包  # torchrun命令会自动启动多个进程，以有效利用多个GPU

!torchrun \
    --nproc_per_node 2 pytorch/summarization/run_summarization.py \
    --fp16 \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /kaggle/workspace/output/distributed_training/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

![image](https://github.com/user-attachments/assets/213fb054-64e9-4031-a9e2-7437ea0dbaf5)


### 使用accelerate来分布式微调
```shell
# accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 {script_name.py} {--arg1} {--arg2} ...

!accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 run_summarization_no_trainer.py \
    --model_name_or_path malmarjeh/mbert2mbert-arabic-text-summarization \
    --dataset_name autoevaluate/autoeval-staging-eval-project-cnn_dailymail-899c0b5b-10935468 \
    --dataset_config "default" \
    --source_prefix "summarize: " \
    --output_dir /kaggle/workspace/output_accelerate/tst-summarization
```

#### 由于cnn-dailymail数据集过大，我们使用的 2x NVIDIA T4 显存不够， 在做数据集tokenize的时候，只能加载到70%就爆显存（超时错误）


#### 更换数据集为 `autoevaluate/autoeval-staging-eval-project-cnn_dailymail-899c0b5b-10935468`后，虽然尺寸变小了，但是缺少验证集，同样报错


---

### 使用自定义数据集来分布式微调T5

#### 我们从huggingface上拉取了一个bbc_news数据集，并使用gensim.summarization模块+datasets库，把bbc_news转换成了和cnn_dailymail相同的格式
```shell
!python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --train_file /kaggle/working/dataset/bbc_news_formatted_train.json \
    --validation_file /kaggle/working/dataset/bbc_news_formatted_test.json \
    --text_column "article" \
    --summary_column "highlights" \
    --source_prefix "summarize: " \
    --output_dir /kaggle/workspace/accelerate1/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```
#### 训练开始
![image](https://github.com/user-attachments/assets/930a5ce9-bf48-40c1-bd57-8d48b18080d2)

#### 训练结果+评测结果
![image](https://github.com/user-attachments/assets/c3134ac1-bd85-488b-bd64-46d105f9f6d4)


### 由于之前的CNN-dailymail数据集太大，我们将其截断，继续train
```shell
# 截断选项
  --max_train_samples 50 \
  --max_eval_samples 50 \
  --max_predict_samples 50 \
```

```shell
!python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --max_train_samples 50 \
    --max_eval_samples 50 \
    --max_predict_samples 50 \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /kaggle/workspace/truncation_train/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

### 训练结果
-中途不小心打断了，没测完~~~
- 但是基本训练完了
- ![image](https://github.com/user-attachments/assets/2a98d6f6-5910-4c11-b982-cfcd7bc76949)


### 从检查点加载训练到一半的模型，继续训练
```shell
!python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --max_train_samples 100 \
    --max_eval_samples 100 \
    --max_predict_samples 100 \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /kaggle/workspace/resume_train/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --resume_from_checkpoint /kaggle/workspace/truncation_train/tst-summarization \
    --predict_with_generate
```

### 训练结果+评测结果
![image](https://github.com/user-attachments/assets/b3eeb903-401a-4d90-ac61-d21259b2f5eb)

