import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from setting.dataset import Interview_Dataset
from utils.file_utils import unzip_files


class Controller:
    def __init__(self):
        """
        Controller 클래스의 생성자.
        Hugging Face 토큰을 .env 파일에서 불러옴.
        """
        self.token = self._load_environment()

    @staticmethod
    def _load_environment():
        """
        환경 변수 로드 및 Hugging Face 토큰 가져오기
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, ".env")
        load_dotenv(dotenv_path)

        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("Hugging Face 토큰을 .env 파일에 설정해주세요. 예: HUGGINGFACE_TOKEN=your_token")
        return token

    def main(self, input_dir="./02.라벨링데이터", output_file="processed_interview_data_valid.jsonl"):
        """
        데이터 로드, 처리, 저장을 순차적으로 실행하는 함수

        :param input_dir: 압축 해제된 JSON 파일이 저장된 디렉토리
        :param output_file: 처리된 데이터가 저장될 JSONL 파일 경로
        """
        # ZIP 파일 압축 해제
        unzip_files(input_dir)

        # 하나의 Interview_Dataset 인스턴스를 생성하여 모든 데이터를 누적
        dataset = Interview_Dataset(input_file=None)  # 파일 경로는 나중에 전달

        # 압축 해제된 디렉토리 내의 모든 JSON 파일을 처리
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)

                    # 데이터 로드
                    raw_data = dataset.load_data(file_path)

                    # 데이터 처리
                    dataset.process_data(raw_data)

        # 모든 데이터를 하나의 JSONL 파일로 저장
        dataset.save_to_file(dataset.processed_data, output_file)

    @classmethod
    def push_to_hub(cls, output_file: str, dataset_name: str, token: str):
        """
        Hugging Face 데이터셋 허브에 업로드
        """
        login(token=token)
        dataset = load_dataset("json", data_files=output_file)
        dataset.push_to_hub(dataset_name)


if __name__ == '__main__':
    controller = Controller()
    controller.main()
    controller.push_to_hub("processed_interview_data_valid.jsonl", "UICHEOL-HWANG/InterView_Datasets_Val", controller.token)