import wandb
import os


class TrackingTrain:
    @staticmethod
    def initialize(project_name, run_name=None, config=None, entity="icucheol"):
        """
        WandB 초기화

        :param project_name: WandB 프로젝트 이름
        :param run_name: 실행 이름
        :param config: 학습 설정
        :param entity: WandB 사용자 엔터티
        """
        wandb.login()
        id = wandb.util.generate_id()  # 고유 ID 생성
        wandb.init(
            project=project_name,
            id=id,
            name=run_name,
            entity=entity,
            config=config,
        )
        print(f"WandB 프로젝트 '{project_name}' 초기화 완료. 실행 ID: {id}")

    @staticmethod
    def log_metrics(metrics):
        """
        WandB에 메트릭 기록

        :param metrics: 기록할 메트릭 딕셔너리
        """
        if not isinstance(metrics, dict):
            raise ValueError("metrics는 딕셔너리 형태여야 합니다.")
        wandb.log(metrics)
        print(f"WandB에 메트릭 기록 완료: {metrics}")

    @staticmethod
    def log_model_path(save_path):
        """
        WandB에 모델 저장 경로 기록 및 아티팩트 업로드

        :param save_path: 모델 저장 디렉토리
        """
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"경로 '{save_path}'가 존재하지 않습니다.")

        # 모델 경로 로깅
        wandb.log({"model_save_path": save_path})
        print(f"모델 저장 경로 '{save_path}'를 WandB에 기록했습니다.")

        # 모델 파일 업로드
        artifact = wandb.Artifact("trained_model", type="model")
        artifact.add_dir(save_path)
        wandb.log_artifact(artifact)
        print(f"모델 '{save_path}'를 WandB에 아티팩트로 업로드 완료.")

    @staticmethod
    def finish_wandb():


        wandb.finish()
        print("WandB 세션 종료 완료")