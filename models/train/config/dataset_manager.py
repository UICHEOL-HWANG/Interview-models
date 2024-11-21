from datasets import load_dataset

class DatasetManager:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def load_dataset(repository_id: str, split="train", sample_fraction=None):
        """
        데이터셋 로드 및 샘플링
        :param repository_id: 데이터셋 경로
        :param split: 데이터 분류 (train, validation 등)
        :param sample_fraction: 데이터 샘플링 비율 (훈련 데이터셋에만 적용)
        :return: datasets 객체
        """
        print(f"데이터셋 로드: {repository_id} ({split})")
        dataset = load_dataset(repository_id, split=split)

        # Train <-> Validation 구분
        if sample_fraction:
            dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * sample_fraction)))
        return dataset # Train은 68,000 rows -> 40%만 추출 해서 학습 예정

    def preprocess_dataset(self, dataset):
        """
        데이터 전처리: 직업, 나이, 경력, 질문, 답변을 포함한 입력 구성
        """
        special_tokens = {
            "inst_token": "<INST>",
            "inst_end_token": "<INST_END>",
        }

        def combine_texts(example):
            combined_text = (
                f"{special_tokens['inst_token']} "
                f"직업: {example['occupation']}, 나이 범위: {example['ageRange']}, 경력: {example['experience']} "
                f"질문: {example['question']} 답변: {example['answer']} "
                f"{special_tokens['inst_end_token']} {self.tokenizer.eos_token}"
            )
            return {"text": combined_text}

        return dataset.map(combine_texts)