import argparse
from config.model_manager import ModelManager
from config.training_manager import TrainingManager
from config.dataset_manager import DatasetManager


def main(args):
    # ModelManager 초기화 및 모델, 토크나이저 로드
    model_manager = ModelManager(base_model=args.base_model)
    model, tokenizer = model_manager._load_model_and_tokenizer()


    # DatasetManager 초기화 및 데이터셋 로드
    dataset_manager = DatasetManager()
    dataset = dataset_manager.load_dataset(args.dataset_repo, split="train")
    dataset = dataset_manager.preprocess_dataset(dataset)

    # TrainingManager 초기화 및 훈련 파라미터 구성
    training_manager = TrainingManager()
    peft_config = training_manager.configure_peft()
    training_args = training_manager.configure_training(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate
    )

    # 모델 학습
    training_manager.train_model(model, tokenizer, dataset, peft_config, training_args)


if __name__ == "__main__":
    # argparse로 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Train a KoAlpaca model with QLoRA")

    # 기본 모델 설정
    parser.add_argument(
        "--base_model",
        type=str,
        default="beomi/KoAlpaca-Polyglot-5.8B",
        help="베이스 모델 이름"
    )

    # 데이터셋 설정
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default="UICHEOL-HWANG/InterView_Datasets",
        help="Hugging Face 데이터셋 저장소 ID"
    )

    # 출력 디렉토리 설정
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results",
        help="모델 저장 디렉토리"
    )

    # 학습 관련 설정
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="훈련 에폭 수"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="학습률"
    )

    # 인자 파싱 및 main 함수 실행
    args = parser.parse_args()
    main(args)