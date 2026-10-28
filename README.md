# Imagen

**Apple Silicon Mac에서 FLUX 3 Image Klein을 로컬로 실행하는 이미지 생성·편집 도구입니다.**

텍스트 생성, 여러 사진을 이용한 편집, 중요한 부분의 자동 디테일 참조를 지원합니다. 기본 출력은 512×512이며, 모든 생성 입력도 각각 512 이하로 준비합니다. 모델은 기존 Q8 체크포인트를 사용하고 실행 중 CPU·GPU 연산과 메모리 배치를 최적화합니다.

[English](README.en.md) · [사용법](docs/usage.md) · [설계](docs/architecture.md) · [벤치마크](docs/benchmarks.md) · [개발·검증](docs/development.md)

## 한눈에 보기

- **로컬 생성:** 사진을 외부 생성 API에 보내지 않고 설치된 MFLUX·MLX로 실행합니다.
- **자동 디테일 참조:** 얼굴, 글자, 눈에 띄는 물체를 찾아 원본에서 먼저 잘라내고 별도 참조로 전달합니다.
- **크기 제한:** 출력은 최대 512×512. 입력은 비율을 유지하며 각각 긴 변 512 이하. 작은 이미지는 확대하지 않습니다.
- **계산량 제한:** 자동 세부 참조는 최대 3개, 추가 예산 1,536토큰. 전체 참조는 최대 10장입니다.
- **C++ 디노이징:** 트랜스포머와 스케줄러는 MLX C++ API로 실행합니다. 텍스트·참조 인코딩과 이미지 디코딩은 MFLUX Python을 사용합니다. 인코딩과 디코딩의 C++ 전환은 아직 완료되지 않았습니다.

## 성능 측정

Apple M3 Pro / 36GB에서 같은 512×352 편집 입력의 16스텝 디노이징은 기존 C++ 경로 72.81초, 최적화 경로 64.60초였습니다. 각 1회이며 모델 로딩·인코딩·디코딩·저장은 제외합니다. 최종 잠재값은 해당 Python 기준과 비트 일치했습니다. 50초 목표는 아직 달성하지 못했습니다.

이는 한 입력과 한 기기의 기록입니다. 적용한 최적화, 실험 조건과 제외한 시도는 [최적화 실험 정리](docs/optimization-experiments.md)에 있습니다.

## 시작하기

### 1. 준비

검증 환경은 **M3 Pro / 36GB / macOS 27.0.1 / Python 3.12 / MFLUX 0.24.0 / MLX 0.32.4**입니다. 다른 칩·메모리 구성은 별도로 측정하지 않았습니다.

필요한 도구:

- Apple Silicon Mac과 MFLUX·MLX가 지원하는 macOS
- [Git](https://git-scm.com/), [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Xcode Command Line Tools: `xcode-select --install` — 자동 영역 검출에 Swift·Vision을 사용합니다.
- Hugging Face 계정 및 모델 접근 승인. 모델은 여러 GB이며 Q8 변환 중 원본 다운로드 캐시도 필요합니다.

```sh
git clone https://github.com/corip-sm/Imagen.git
cd Imagen
./setup.sh
```

`setup.sh`는 [검증한 MFLUX 커밋](https://github.com/mflux-community/mflux/commit/8a7fd0a486635f670ef90e02efe1c051ed3f133c)과 MLX·MLX Metal 0.32.4를 설치하고 C++ 디노이저를 빌드합니다. 모델 파일은 이 명령에 포함되지 않습니다. 명령이 안 잡히면 `uv tool update-shell` 후 터미널을 다시 열어주세요.

### 2. 모델 준비 — 최초 한 번

[FLUX 3 Image Klein 모델 페이지](https://huggingface.co/black-forest-labs/flux-3-image-klein)에서 접근 조건에 동의하고, 해당 저장소를 읽을 수 있는 Hugging Face 토큰을 준비합니다. 다음 로그인 명령은 토큰을 숨김 입력으로 받습니다.

```sh
uv run --no-project --python 3.12 --with huggingface-hub python -c \
  'import getpass; from huggingface_hub import login; login(token=getpass.getpass("HF read token: "), add_to_git_credential=False)'

export IMAGEN_MODEL="$HOME/.cache/imagen/flux3-klein-q8"
mflux-save --model flux3-image-klein --quantize 8 --path "$IMAGEN_MODEL"
```

이미 Q8 모델이 있으면 다운로드·변환 대신 `export IMAGEN_MODEL="/path/to/existing-q8"`만 지정합니다. 기존 로컬 프로젝트의 `~/.cache/gaon-image/flux3-klein-q8`도 자동으로 찾습니다.

### 3. 텍스트로 생성

영어로 장면·피사체·조명·구도를 자세히 적습니다. 이 실행기는 별도 프롬프트 확장 모델을 자동 실행하지 않습니다.

```sh
./generate.sh outputs/cafe.png \
  "A realistic photograph of a quiet neighborhood cafe at dawn. Warm light glows through the windows onto a wet stone sidewalk. A small wooden table holds a white ceramic cup. Eye-level composition, natural colors, soft daylight, detailed materials, no text or watermark."
```

### 4. 사진 편집

```sh
./generate.sh outputs/edited.png \
  "A realistic photograph of the same room with the dog completely removed. The blue fabric chair, white bedding, furniture and lighting remain consistent with the reference. Naturally reconstruct the fabric and bedding previously hidden by the dog, preserving the original camera angle and composition." \
  /path/to/photo.png
```

원본을 수정하지 않고 `outputs/edited.png`를 만듭니다. 자동으로 전체 축소본과 중요한 부분을 준비하고, 결과 옆 `edited.references/`에 전달한 참조와 선택 기록을 저장합니다. 정밀 편집은 주변 영역도 달라질 수 있으며 마스크 기반 원본 픽셀 보존을 보장하지 않습니다.

### 5. 더 작은 크기·세로 이미지

```sh
SIZE=256 ./generate.sh outputs/small.png "A detailed English scene caption ..."
WIDTH=384 HEIGHT=512 ./generate.sh outputs/portrait.png \
  "A detailed English edit instruction ..." /path/to/photo.png
```

크기는 16의 배수입니다. 512 이내의 유효한 요청은 그대로 사용하고, 큰 요청은 비율을 유지해 긴 변 512로 줄여 모델 단위에 맞춰 내림합니다. `SIZE`보다 `WIDTH`·`HEIGHT`가 우선합니다.

## 어떻게 최적화하나요?

1. 큰 스트림의 Q8 가중치를 메모리에서 한 번 BF16로 복원해 반복 복원을 줄입니다.
2. 큰 MLP 입력 투영의 출력 채널 약 45%는 CPU FP32, 나머지는 GPU BF16로 나눠 계산합니다. 내적 중간 합을 나누지 않습니다.
3. 큰 스트림의 Q/K/V 가중치를 결합하고, 요청 안에서 바뀌지 않는 텍스트·참조 투영과 위치 임베딩을 재사용합니다.
4. 원본 LayerNorm, 반올림 단계와 합산 순서를 유지합니다. 작은 스트림은 Q8 연산을 사용합니다.
5. 참조 사진은 전체 구도와 중요한 세부 영역을 분리해 불필요한 고해상도 배경 계산을 줄입니다.

C++ 실행기는 큰 스트림에서 정규화·RoPE·활성화·잔차 연산을 기존 Metal 커널로 융합하고 독립적인 attention과 MLP 작업을 함께 제출합니다. 마지막 블록에서는 전체 참조를 attention에 사용하면서 이미지 출력에 필요한 토큰만 계산합니다. C++ 전환 자체가 속도나 품질을 개선한다는 보장은 없습니다.

실행기 최적화와 참조 선택은 별도 계층입니다. 전자는 같은 입력의 수치 비교 대상이고, 후자는 모델에 전달하는 입력을 바꿉니다. [상세 설계](docs/architecture.md) · [적용·제외한 실험](docs/optimization.md)

## 설정과 문제 해결

| 설정 | 기본값 | 용도 |
|---|---|---|
| `IMAGEN_MODEL` | `~/.cache/imagen/flux3-klein-q8` | 기존 로컬 Q8 모델 위치 |
| `SIZE` | `512` | 정사각형 출력 |
| `WIDTH`, `HEIGHT` | `SIZE` | 직사각형 출력 |
| `STEPS`, `SEED` | `16`, `7` | 스텝·시드 |
| `REFERENCE_REGIONS` | 자동 선택 | 직접 영역 지정 JSON. `[]` 파일은 전체 참조만 사용 |
| `IMAGEN_RUNTIME` | `1` | `0`이면 실행기 최적화만 끔. 512 정책·참조 처리는 유지 |

모델을 못 찾거나, Swift가 없거나, 자동 선택이 편집 대상과 다르면 [사용법·문제 해결](docs/usage.md)을 참고하세요. 물체 선택은 시각적 중요도 기반이며, 상품 종류나 편집 의도를 완전히 이해하는 기능은 아닙니다.

## 프로젝트 구조

```text
Imagen/
├── generate.sh                 # 기본 생성·편집 명령
├── setup.sh                    # 설치 진입점
├── src/imagen/
│   ├── references/             # 크기 정책·자동 중요 영역·CLI 실행
│   ├── native/                 # C++ 디노이저와 Python 입력·디코딩 연결
├── scripts/                    # 설치, Python 실행, 벤치마크, 그래프
├── tests/                      # 모델 없이 실행하는 회귀 검사
├── benchmarks/data/            # 공개용 숫자 측정값
├── docs/assets/                # README용 재생성 가능한 SVG 그래프
├── examples/                   # 수동 영역 JSON 예시
└── outputs/                    # 생성 결과·참조: Git에서 제외
```

[개발·검증](docs/development.md) · [기여 안내](CONTRIBUTING.md)

## 라이선스·출처

소프트웨어는 [Apache License 2.0](LICENSE)을 따릅니다. 참고한 런타임·MLX 구현의 출처와 MIT 고지는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 보존합니다. 모델 가중치는 포함하지 않으며 별도의 [FLUX Kommunity License와 접근 조건](https://huggingface.co/black-forest-labs/flux-3-image-klein)을 따릅니다. 모델 사용 권한은 이 저장소의 소프트웨어 라이선스와 별개입니다.
