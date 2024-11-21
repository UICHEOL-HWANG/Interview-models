import wandb

class TrackingTrain:
    @staticmethod
    def initialize(project_name, run_name=None, config=None, entity="icucheol"):
        """
        WandB 초기화

        :param project_name: 완디비 프로젝트명
        :param run_name: 현재 실행 트래킹 명
        :param config: 훈련 설정(딕셔너리 형태)
        :param entity: WandB 사용자 엔터티 (기본값: 'icucheol')
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
    def finish_wandb():
        """
        WandB 세션 종료
        """
        wandb.finish()
        print("WandB 세션 종료 완료")