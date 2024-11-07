import json
from collections import defaultdict
from typing import List, Dict

class Interview_Dataset:
    def __init__(self, input_file: str):
        """
        데이터셋 세팅 전 규합하는 생성자 아규먼츠

        :param input_file: AI Hub input 파일 매개변수 (파일 경로)
        """
        self.input_file = input_file
        self.processed_data = defaultdict(list)

    @classmethod
    def load_data(cls, input_file: str) -> List[Dict]:
        """
        데이터를 로드하여 반환하는 클래스 메서드

        :param input_file: JSON 파일 경로
        :return: JSON 파일의 데이터
        """
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @classmethod
    def save_to_file(cls, data: Dict[str, List[Dict]], output_file: str):
        """
        데이터를 JSONL 파일 형식으로 저장하는 클래스 메서드

        :param data: occupation별로 그룹화된 데이터
        :param output_file: JSONL 파일 경로
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for occupation, entries in data.items():
                for entry in entries:
                    json.dump(entry, f)
                    f.write("\n")

    def process_data(self, data: List[Dict]):
        """
        데이터를 occupation별로 그룹화하여 저장하는 메서드

        :param data: JSON 형식의 데이터
        """
        for entry in data:
            info = entry["dataSet"]["info"]
            question = entry["dataSet"]["question"]["raw"]["text"]
            answer = entry["dataSet"]["answer"]["raw"]["text"]
            # Structure the data
            processed_entry = {
                "experience": info["experience"],
                "ageRange": info["ageRange"],
                "occupation": info["occupation"],
                "question": question,
                "answer": answer
            }

            # Group data by occupation
            occupation = info["occupation"]
            self.processed_data[occupation].append(processed_entry)