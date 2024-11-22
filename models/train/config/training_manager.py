from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tracking.wandb import TrackingTrain

class TrainingManager:
    def __init__(self, project_name, run_name=None):
        self.project_name = project_name
        self.run_name = run_name

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
            report_to="wandb"
        )

    @staticmethod
    def compute_metrics(pred):
        """
        메트릭 계산: 정확도, 정밀도, 재현율, F1 점수
        """
        labels = pred.label_ids
        logits = pred.predictions
        preds = logits.argmax(-1)  # 예측된 클래스

        # 다중 클래스의 경우 평균 방식으로 "macro" 사용
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def train_model(self, model, tokenizer, train_dataset, eval_dataset, peft_config, training_args):
        # WandB 초기화
        TrackingTrain.initialize(
            project_name=self.project_name,
            run_name=self.run_name,
            config=training_args.to_dict()
        )

        try:
            # SFTTrainer 초기화
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
                tokenizer=tokenizer,
                args=training_args,
                packing=False,
                max_seq_length=512,  # max_seq_length 직접 전달
                dataset_text_field="text",  # dataset_text_field 직접 전달
                compute_metrics=self.compute_metrics, # Validation Test

            )

            # 학습 시작
            print(f"훈련 시작: 저장 경로 -> {training_args.output_dir}")
            trainer.train()

            # 평가
            if eval_dataset:
                metrics = trainer.evaluate()
                print(f"검증 결과: {metrics}")

            # 모델 저장
            save_path = os.path.join(training_args.output_dir, "UICHEOL-HWANG/KoAlpaca-InterView-5.8B")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            trainer.save_model(save_path)
            print(f"{save_path} 경로로 저장 완료")

        except Exception as e:
            print(f"훈련 중 오류 발생: {e}")

        finally:
            TrackingTrain.finish_wandb()