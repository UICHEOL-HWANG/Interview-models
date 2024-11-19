import json
from typing import List, Dict
from collections import defaultdict

class Interview_Dataset:
    def __init__(self, input_file: str):
        """
        데이터셋 초기화

        :param input_file: AI Hub input 파일 경로
        """
        self.input_file = input_file
        self.processed_data = defaultdict(list)  # occupation별로 그룹화된 데이터 저장

    @classmethod
    def load_data(cls, input_file: str) -> Dict:
        """
        데이터를 로드하여 반환하는 클래스 메서드

        :param input_file: JSON 파일 경로
        :return: JSON 파일의 데이터
        """
        try:
            with open(input_file, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Error: {input_file} 파일이 존재하지 않습니다.")
            return None
        except json.JSONDecodeError:
            print(f"Error: {input_file}는 유효한 JSON 형식이 아닙니다.")
            return None
        except Exception as e:
            print(f"Error: {input_file}를 로드하는 중 오류 발생: {e}")
            return None
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

    def process_data(self, data: Dict):
        """
        데이터를 occupation별로 그룹화하여 저장하는 메서드

        :param data: JSON 형식의 데이터
        """
        if not isinstance(data, dict):
            print("Error: 데이터는 JSON 형식이어야 합니다.")
            return

        # 필요한 데이터 추출 및 구조화
        info = data["dataSet"]["info"]
        question = data["dataSet"]["question"]["raw"]["text"]
        answer = data["dataSet"]["answer"]["raw"]["text"]

        processed_entry = {
            "experience": info["experience"],
            "ageRange": info["ageRange"],
            "occupation": info["occupation"],
            "question": question,
            "answer": answer
        }

        # occupation에 따라 데이터 그룹화
        occupation = info["occupation"]
        self.processed_data[occupation].append(processed_entry)