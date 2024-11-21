from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

class TrainingManager:
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
            report_to="tensorboard"
        )

    def train_model(self, model, tokenizer, dataset, peft_config, training_args):
        """
        모델 학습
        """
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_args,
            packing=False,
        )
        trainer.train()
        save_path = f"{training_args.output_dir}/UICHEOL-HWANG/KoAlpaca-InterView-5.8B"
        trainer.save_model(save_path)
        print(f"{save_path} 경로로 저장 완료")