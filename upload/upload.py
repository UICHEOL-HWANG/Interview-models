from dotenv import load_dotenv
import os
from huggingface_hub import login

class ModelUploader:
    # .env 파일에서 Hugging Face API 토큰 가져오기
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, ".env")
        load_dotenv(dotenv_path)
        self.model = model
        self.tokenizer = tokenizer

        self.token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            raise ValueError("Hugging Face 토큰을 .env 파일에 설정해주세요. 예: HUGGINGFACE_TOKEN=your_token")
    def login_huggingface(self):
        login(token=self.token)
        print("Hugging Face 허브에 로그인 완료!")

    def push_model_to_hub(self, repository_id="UICHEOL-HWANG/Interview-model"):
        """
        모델 및 토크나이저를 Hugging Face 허브에 업로드
        """
        print(f"모델을 Hugging Face 허브에 업로드 중: {repository_id}")
        self.model.push_to_hub(repository_id)
        self.tokenizer.push_to_hub(repository_id)
        print("모델과 토크나이저 업로드 완료!")

if __name__ == "__main__":
    uploader = ModelUploader()
    uploader.login_huggingface()
    uploader.push_model_to_hub()