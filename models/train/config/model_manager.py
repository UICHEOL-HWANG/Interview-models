from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import os


class ModelManager:
    def __init__(self, base_model="beomi/KoAlpaca-Polyglot-5.8B"):
        """
        ModelManager 초기화
        """
        self.base_model = base_model
        self.token = self._load_environment()
        self.attn_implementation, self.torch_dtype = self._configure_hardware()

    def _configure_hardware(self):
        """
        GPU 기능에 따라 Flash Attention 및 dtype 설정
        """
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            print("Flash Attention 2 및 bfloat16 활성화")
            os.system("pip install -qqq flash-attn")  # Flash Attention 설치
            return "flash_attention_2", torch.bfloat16
        else:
            print("Eager Attention 및 float16 사용")
            return "eager", torch.float16

    @staticmethod
    def _load_environment():
        """
        환경 변수 로드 및 Hugging Face 토큰 로드
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, ".env")
        load_dotenv(dotenv_path)

        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("Hugging Face 토큰을 .env 파일에 설정해주세요. 예: HUGGINGFACE_TOKEN=your_token")
        return token

    def _login_huggingface(self):
        """
        Hugging Face 허브 로그인
        """
        login(token=self.token)
        print("Hugging Face 허브에 로그인 완료!")

    def _load_model_and_tokenizer(self):
        """
        모델과 토크나이저 로드
        """
        quant_config = self.configure_quant_config()
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return model, tokenizer

    def configure_quant_config(self):
        """
        양자화 설정
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch_dtype,
            bnb_4bit_use_double_quant=False,
        )


if __name__ == "__main__":
    model_manager = ModelManager()
    model_manager._login_huggingface()
    model, tokenizer = model_manager._load_model_and_tokenizer()