from datasets import load_dataset

class DatasetManager:
    @staticmethod
    def load_dataset(repository_id: str, split="train"):
        """
        :param repository_id: 데이터셋 경로
        :param split: 분류 구분
        :return: datasets Types[jsonl]
        """
        print(f"데이터셋 로드 {repository_id}")
        dataset = load_dataset(repository_id, split=split)
        smaller_dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * 0.4)))
        # 기존 데이터셋은 6만8천줄 json 데이터라 5에폭 기준 훈련 세트가 2만회를 넘기에, 40%의 훈련셋만 세팅

        return smaller_dataset

    @staticmethod
    def preprocess_dataset(dataset):
        """

        :param dataset: load_dataset에서 받아온 데이터를 모델 입력 데이터로 변환
        :return: json
        """

        def combine_texts(example):
            keys = ["experience", "ageRange", "occupation", "question", "answer"]
            combined_text = ", ".join([f"{key}: {example[key]}" for key in keys if key in example])
            return {"text": combined_text}
        return dataset.map(combine_texts)

    