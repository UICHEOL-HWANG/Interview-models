from datasets import load_dataset
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from dotenv import load_dotenv


class InterviewModels:
    def __init__(self, base_model="beomi/Llama-3-Open-Ko-8B"):
        self.base_model = base_model
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.peft_params = self.configure_peft()
        self.training_params = self.configure_training()

        # .env 파일에서 Hugging Face API 토큰 가져오기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, ".env")
        load_dotenv(dotenv_path)

        self.token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            raise ValueError("Hugging Face 토큰을 .env 파일에 설정해주세요. 예: HUGGINGFACE_TOKEN=your_token")

        # Hugging Face 로그인
        self.login_huggingface()

    def login_huggingface(self):
        login(token=self.token)
        print("Hugging Face 허브에 로그인 완료!")

    def _load_model(self):
        quant_config = self.configure_quant_config()
        return AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map="auto"
        )

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    @staticmethod
    def configure_quant_config():
        torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quan=True,
        )

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
    def configure_training():
        return TrainingArguments(
            output_dir="./results",
            num_train_epochs=5,
            per_device_train_batch_size=1,  # 배치 크기를 2에서 1로 줄임
            gradient_accumulation_steps=16,  # 감소한 배치 크기를 보완하기 위해 steps 증가
            optim="paged_adamw_32bit",
            save_steps=50,
            logging_steps=10,
            learning_rate=1e-4,
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

    def preprocess_dataset(self, dataset):
        """
        데이터셋 전처리: 모든 키를 결합하여 텍스트 생성
        """
        def combine_texts(example):
            return {
                "text": (
                    f"경험: {example['experience']}, "
                    f"나이 범위: {example['ageRange']}, "
                    f"직무: {example['occupation']}, "
                    f"질문: {example['question']}, "
                    f"답변: {example['answer']}"
                )
            }

        # 데이터셋에 "text" 필드 추가
        return dataset.map(combine_texts)

    def train_model(self, dataset):
        """
        모델 학습
        """
        dataset = self.preprocess_dataset(dataset)
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            peft_config=self.peft_params,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=self.tokenizer,
            args=self.training_params,
            packing=False,
        )
        trainer.train()


    @staticmethod
    def load_dataset_from_hub(repository_id="UICHEOL-HWANG/InterView_Datasets"):
        """
        Hugging Face 데이터셋 허브에서 데이터셋 로드
        """
        print(f"Hugging Face 데이터셋 로드: {repository_id}")
        return load_dataset(
            repository_id,
            split="train",
        )


if __name__ == "__main__":
    model = InterviewModels()
    dataset = model.load_dataset_from_hub()
    model.train_model(dataset)