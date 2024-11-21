from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os

# Module
from tracking.wandb import TrackingTrain


class TrainingManager:
    def __init__(self, project_name, run_name=None):
        """
        TrainingManager 초기화

        :param project_name: WandB 프로젝트 이름
        :param run_name: 실행 이름
        """
        self.project_name = project_name
        self.run_name = run_name

    @staticmethod
    def configure_peft():
        """
        QLoRA 설정
        """
        return LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

    @staticmethod
    def configure_training(output_dir, num_train_epochs=5, learning_rate=1e-4):
        """
        훈련 파라미터 설정
        """
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            optim="paged_adamw_32bit",
            save_steps=50,
            logging_steps=10,
            learning_rate=learning_rate,
            weight_decay=0.01,
            fp16=True,
            bf16=False,
            max_grad_norm=1.0,
            max_steps=-1,
            warmup_ratio=0.05,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="wandb"
        )

    def initialize_trainer(self, model, tokenizer, dataset, peft_config, training_args):
        """
        SFTTrainer 초기화
        """
        sft_config = SFTConfig(
            max_seq_length=512,
            dataset_text_field="text",
            padding=True,  # 텍스트 패딩
            truncation=True,  # 텍스트 자르기
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
            args=training_args,
            packing=False,
            sft_config=sft_config,
        )
        return trainer

    def train_model(self, model, tokenizer, train_dataset, eval_dataset, peft_config, training_args):
        # WandB 초기화
        TrackingTrain.initialize(
            project_name=self.project_name,
            run_name=self.run_name,
            config=training_args.to_dict()
        )

        try:
            # Trainer 초기화
            trainer = self.initialize_trainer(
                model=model,
                tokenizer=tokenizer,
                dataset=train_dataset,
                peft_config=peft_config,
                training_args=training_args,
            )
            trainer.eval_dataset = eval_dataset  # 검증 데이터셋 추가

            # 학습 시작
            print(f"훈련 시작: 저장 경로 -> {training_args.output_dir}")
            trainer.train()

            # 평가
            if eval_dataset:
                metrics = trainer.evaluate()
                print(f"검증 결과: {metrics}")

            # 모델 저장
            save_path = f"{training_args.output_dir}/UICHEOL-HWANG/KoAlpaca-InterView-5.8B"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            trainer.save_model(save_path)
            print(f"{save_path} 경로로 저장 완료")

        except Exception as e:
            print(f"훈련 중 오류 발생: {e}")

        finally:
            # WandB 세션 종료
            TrackingTrain.finish_wandb()