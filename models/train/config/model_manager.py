import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
from huggingface_hub import login
import torch

class ModelManager:
    def __init__(self, base_model="beomi/KoAlpaca-Polyglot-5.8B"):
        """
        모델 메인 파츠 클래스
        :param base_model: 디폴트 값 base model 준범이 아죠쉬의 base model
        """
        self.base_model = base_model
        self.token = self._load_environment()

    @staticmethod
    def _load_environment():
        """
        환경 변수 로드, 허깅페이스 토큰 로드
        :return:
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, ".env")
        load_dotenv(dotenv_path)

        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("토큰 설정 제대로 해라 .env 파일로 : HUGGINGFACE_TOKEN")
        return token

    def _login_huggingface(self):
        """
        허깅페이스 로그인
        :return:
        """
        login(token=self.token)
        print("허깅페이스 로그인 완료!")

    def _load_model_get_tokenizer(self):
        """
        모델 토크나이저 로드
        :return:
        """
        quant_config = self.configure_quant_config()
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quant_config=quant_config,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return model, tokenizer

    @staticmethod
    def configure_quant_config():
        """
        양자화
        :return:
        """
        torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=False,
        )

if __name__ == "__main__":
    model_manager = ModelManager()