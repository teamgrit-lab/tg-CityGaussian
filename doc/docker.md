## Docker (AWS g6e.12xlarge 권장 설정)

이 문서는 g6e.12xlarge(L40S) 환경에서 GPU를 활용해 학습/추론을 실행할 수 있도록 도커 환경을 구성하는 방법을 설명합니다.

### 1) 사전 준비
- EC2에 NVIDIA 드라이버와 NVIDIA Container Toolkit이 설치되어 있어야 합니다.
- 서브모듈이 필요하다면 호스트에서 먼저 초기화하세요.

```bash
git submodule update --init --recursive
```

### 2) 환경 변수 파일 준비

`docker.env.example`을 복사해 `docker.env`로 만든 뒤 값을 채워주세요.
S3와 MLflow는 나중에 연결할 수 있도록 변수를 비워둬도 됩니다.

```bash
cp docker.env.example docker.env
```

### 3) 이미지 빌드

```bash
docker compose build
```

### 4) 컨테이너 실행

```bash
docker compose run --rm citygs bash
```

컨테이너 안에서는 기존 실행 스크립트를 그대로 사용하면 됩니다.

### 빠른 실행 스크립트(EC2)

```bash
bash scripts/ec2_docker_setup.sh
bash scripts/run_citygs_docker.sh
```

백그라운드 실행(빌드+실행 한 번에):

```bash
bash scripts/launch_citygs.sh
```

실행 명령을 바로 백그라운드로 돌리기:

```bash
bash scripts/run_citygs_job.sh "python main.py fit --config configs/your_config.yaml -n your_run"
```

위 스크립트는 실행 전에 S3에서 데이터를 받고, 실행 후 결과를 업로드합니다.
필요한 값은 `docker.env`에 설정하세요:

```
S3_BUCKET=your-bucket
S3_DATA_PREFIX=datasets
S3_OUTPUT_PREFIX=outputs
```

### 4-1) COLMAP 전용 컨테이너 실행

COLMAP은 별도 인스턴스에서 CLI만 사용할 예정이라면 아래 서비스를 사용하세요.

```bash
docker compose run --rm colmap bash
```

빠른 실행 스크립트:

```bash
bash scripts/run_colmap_docker.sh
```

백그라운드 실행(빌드+실행 한 번에):

```bash
bash scripts/launch_colmap.sh
```

COLMAP 명령을 바로 백그라운드로 돌리기:

```bash
bash scripts/run_colmap_job.sh "colmap help"
```

S3 동기화만 단독으로 실행하려면 컨테이너에서 아래 스크립트를 사용하세요:

```bash
bash scripts/s3_pull.sh
bash scripts/s3_push.sh
```

예시:

```bash
colmap help
```

### 5) S3 연동(예정)

이미지에 `awscli`가 포함되어 있습니다. 데이터/결과 저장 경로를 S3로 연결하는 방식은 아래와 같이 확장할 수 있습니다.

```bash
# 예시: 데이터 다운로드
aws s3 sync s3://your-bucket/datasets /workspace/data

# 예시: 결과 업로드
aws s3 sync /workspace/outputs s3://your-bucket/outputs
```

### 6) MLflow 연동(예정)

`docker.env`에 아래 값을 넣으면 코드에서 `MLFLOW_TRACKING_URI`를 참조해 기록할 수 있습니다.

```
MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
MLFLOW_EXPERIMENT_NAME=citygaussian
```
