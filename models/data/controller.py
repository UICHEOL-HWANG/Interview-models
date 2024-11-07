from Interview_Dataset import Interview_Dataset
from huggingface_hub import login
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
import os




class Controller:
    def __init__(self):
        """
        Controller 클래스의 생성자.
        Hugging Face 토큰을 .env 파일에서 불러옴.
        """
        # 현재 파일이 있는 디렉토리에서 .env 파일 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, ".env")
        load_dotenv(dotenv_path)

        # .env 파일에서 Hugging Face API 토큰 가져오기
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            raise ValueError("Hugging Face 토큰을 .env 파일에 설정해주세요. 예: HUGGINGFACE_TOKEN=your_token")

    def main(self, input_file: str, output_file: str):
        """
        데이터 로드, 처리, 저장을 순차적으로 실행하는 함수

        :param input_file: 원본 JSON 파일 경로
        :param output_file: 처리된 데이터가 저장될 JSONL 파일 경로
        """
        # 데이터셋 객체 생성
        dataset = Interview_Dataset(input_file)

        # 데이터 로드
        raw_data = dataset.load_data(input_file)

        # 데이터 처리
        dataset.process_data(raw_data)

        # 데이터 저장
        dataset.save_to_file(dataset.processed_data, output_file)

    @classmethod
    def push_to_hub(cls, output_file: str, dataset_name: str, token: str):
        login(token=token)
        dataset = load_dataset("json", data_files=output_file)
        dataset.push_to_hub(dataset_name)

if __name__ == '__main__':
    controller = Controller()
    controller.main("interview_data.json", "processed_interview_data.jsonl")
    # 데이터셋 리포지토리는 곧 만들 예정