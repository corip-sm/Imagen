# 사용법과 문제 해결

설치와 최초 모델 준비는 [README](../README.md)를 따른다.

## 기본 명령

```sh
./generate.sh OUTPUT.png "Detailed English caption" [REFERENCE.png ...]
./generate.sh --help
```

텍스트만 넣으면 새 이미지를 생성한다. 참조를 넣으면 편집 CLI를 사용한다. `generate-fast-edit.sh`도 같은 공통 경로를 사용하는 호환 명령이다.

```sh
SIZE=256 ./generate.sh outputs/small.png "A detailed English scene caption ..."
WIDTH=384 HEIGHT=512 SEED=12 ./generate.sh outputs/portrait.png \
  "A full caption describing the desired edited scene ..." /path/to/input.png
```

- 출력: 기본·최대 512×512. 더 작은 16배수 크기는 그대로 사용한다.
- 큰 출력 요청: 긴 변이 512가 되도록 비율을 유지해 줄이고 16의 배수로 내림한다. 1024×768 요청은 512×384가 된다.
- 입력: 방향 정보를 반영한 뒤 각 이미지의 긴 변을 512 이하로 제한한다. 작은 사진은 확대하지 않으며 16배수 정렬은 여백으로 처리한다.
- 입력 순서: 전체 구도 참조가 먼저, 자동 또는 수동 디테일 참조가 그 뒤다. 합계 최대 10장이다.
- 원본을 출력 파일로 지정하는 경우나 준비 파일로 덮어쓸 수 있는 경로는 거부한다.

## 모델 위치

우선순위는 `IMAGEN_MODEL`, 기존 호환 키 `GAON_IMAGE_MODEL`, `~/.cache/imagen/flux3-klein-q8`다. 새 기본 위치가 없고 이전 `~/.cache/gaon-image/flux3-klein-q8`가 있으면 이전 위치를 사용한다.

생성 명령은 오프라인 모드로 실행한다. 최초 다운로드와 인증은 README의 별도 준비 단계다.

## 자동·수동 디테일

기본은 [자동 중요 영역 선택](automatic-references.md)이다. 자동 선택이 편집 대상과 맞지 않으면 직접 영역을 지정할 수 있다.

```json
[{"source": 1, "box": [100, 80, 500, 480]}]
```

`source`는 입력 순서의 1부터 시작하는 번호, `box`는 방향 보정된 원본의 `[left, top, right, bottom]` 픽셀 좌표다. 원본에서 먼저 잘라낸 뒤 512 제한을 적용한다.

```sh
REFERENCE_REGIONS=examples/manual-regions.json ./generate.sh \
  outputs/manual.png "A detailed English edit caption ..." /path/to/input.png

REFERENCE_REGIONS=examples/overview-only.json ./generate.sh \
  outputs/overview.png "A detailed English edit caption ..." /path/to/input.png
```

파일을 지정하면 자동 선택을 대체한다. 빈 목록 `[]`은 디테일 참조를 추가하지 않는다. 전체 축소만으로 작은 얼굴·문자 정보가 손실될 수 있다는 점은 [사례](benchmarks.md)에 기록했다.

## 저장 결과

`outputs/edited.png`를 만들면 `outputs/edited.references/`에 준비된 PNG와 `references.json`이 남는다. 기록에는 원본 크기·선택 좌표·최종 크기·참조 토큰·자동 분석 결과가 포함된다. 원본 경로와 해시도 포함되므로 개인 데이터로 취급한다. 이 폴더와 생성 결과는 Git에서 제외한다.

## 문제 해결

| 증상 | 확인 |
|---|---|
| `mflux-generate-flux3`를 찾지 못함 | `./setup.sh`, `uv tool update-shell`, 새 터미널 |
| 모델 디렉터리를 찾지 못함 | `IMAGEN_MODEL`이 변환된 Q8 모델 폴더인지 확인 |
| 모델 다운로드 권한 오류 | 공식 모델 페이지 접근 승인과 해당 저장소 읽기 토큰 확인 |
| Swift/Vision 분석 실패 | `xcrun --find swift`; Xcode Command Line Tools 설치 상태 확인 |
| 크기 오류 | 512 이하의 16배수 출력 사용: 256, 384, 512 등 |
| 얼굴·글자가 뭉개짐 | 자동 선택 기록 확인 후 중요한 부분을 직접 지정; 입력에 없는 정보는 복구 보장 없음 |
| 메모리가 부족함 | 다른 작업의 메모리 사용 확인; `IMAGEN_RUNTIME=0`으로 추가 메모리 최적화 경로를 끌 수 있음. 지원 메모리 하한은 별도 검증하지 않음 |
| 다른 Mac에서 더 느림 | CPU/GPU 분배는 M3 Pro 실측으로 선택한 값. 다른 칩의 가속은 보장하지 않음 |

정밀 수정도 생성 작업이다. 눈만 바꾸는 지시 등에서 주변 픽셀이 바뀔 수 있고, OCR 참조가 정확한 문자 출력을 보장하지 않는다. 이 도구는 마스크 합성이나 업스케일러를 별도로 제공하지 않는다.
