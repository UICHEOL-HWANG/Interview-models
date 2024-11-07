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
import huggingface_hub


class InterviewModels:
    def __init__(self, base_model="beomi/Llama-3-Open-Ko-8B"):
        self.base_model = base_model
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.peft_params = self.configure_peft()
        self.training_params = self.configure_training()

    @classmethod
    def login_huggingface(cls):
        # Hugging Face login
        huggingface_hub.login()

    @classmethod
    def _install_flash_attention(cls):
        # Install flash attention if required
        if torch.cuda.get_device_capability()[0] >= 8:
            os.system("pip install -qqq flash-attn")

    @classmethod
    def configure_quant_config(cls):
        # QLoRA configuration
        if torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=False,
        )

    def _load_model(self):
        quant_config = self.configure_quant_config()
        return AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map="auto"  # Automatically maps the model to the available device
        )

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    @classmethod
    def configure_peft(cls):
        # PEFT configuration using LoraConfig
        return LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

    @classmethod
    def configure_training(cls):
        # Training configuration using TrainingArguments
        return TrainingArguments(
            output_dir="./results",
            num_train_epochs=10,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=25,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="tensorboard"
        )

    def train_model(self, dataset):
        # SFTT 트레이너 파인튜닝을 위한 아직 Dataset은 세팅하지 않음
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            peft_config=self.peft_params,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=self.tokenizer,
            args=self.training_params,
            packing=False,
        )
        trainer.train()
