from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import os
from evaluate import load
from tracking.wandb import TrackingTrain


class TrainingManager:
    def __init__(self, project_name, run_name=None, tokenizer=None):
        self.project_name = project_name
        self.run_name = run_name
        self.tokenizer = tokenizer

    @staticmethod
    def configure_peft():
        return LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

    @staticmethod
    def configure_training(output_dir, num_train_epochs=5, learning_rate=1e-4):
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=32,
            optim="paged_adamw_32bit",
            save_steps=50,
            logging_steps=10,
            learning_rate=learning_rate,
            weight_decay=0.02,
            fp16=True,
            bf16=False,
            max_grad_norm=1.0,
            max_steps=-1,
            warmup_ratio=0.05,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="wandb",
        )

    def compute_metrics(self, pred):
        if not self.tokenizer:
            raise ValueError("tokenizer가 None입니다.")
        metric = load("rouge")
        labels = pred.label_ids
        logits = pred.predictions
        decoded_preds = [self.tokenizer.decode(p, skip_special_tokens=True) for p in logits]
        decoded_labels = [self.tokenizer.decode(l, skip_special_tokens=True) for l in labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {
            "rouge1": result["rouge1"].mid.fmeasure,
            "rouge2": result["rouge2"].mid.fmeasure,
            "rougeL": result["rougeL"].mid.fmeasure,
        }

    def train_model(self, model, tokenizer, train_dataset, eval_dataset, peft_config, training_args):
        # WandB 초기화
        TrackingTrain.initialize(self.project_name, self.run_name, training_args.to_dict())
        try:
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
                tokenizer=tokenizer,
                args=training_args,
                packing=False,
                max_seq_length=512,
                dataset_text_field="text",
                compute_metrics=self.compute_metrics,
            )

            # 학습 시작
            print(f"훈련 시작: 저장 경로 -> {training_args.output_dir}")
            trainer.train()

            # 평가
            if eval_dataset:
                metrics = trainer.evaluate()
                TrackingTrain.log_metrics(metrics)  # 메트릭 기록
                print(f"검증 결과: {metrics}")

            # 모델 저장
            save_path = os.path.join(training_args.output_dir, "UICHEOL-HWANG/KoAlpaca-InterView-5.8B")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            trainer.save_model(save_path)
            TrackingTrain.log_model_path(save_path)  # 모델 경로 로깅 및 업로드
            print(f"{save_path} 경로로 모델 저장 완료")

        except Exception as e:
            print(f"훈련 중 오류 발생: {e}")

        finally:
            TrackingTrain.finish_wandb()