# Unit Test 구성 및 2D → 3D 변환 실습
## 1. 과제 목적

Unit Test를 통한 코드 검증과 2D 이미지를 3D 데이터로 변환하는 기본적인 Depth Map 기반 파이프라인을 구현하는 것을 목표로 한다.

## 2. 주요 구현 내용
### (1) Unit Test 구성
- pytest를 사용하여 Depth Map 생성 함수 검증

- 테스트 항목
  
정상 입력 시 출력 shape 및 타입 확인

입력 이미지가 없을 경우 예외 처리 확인

### (2) 2D → Depth Map → 3D 데이터 변환

- Grayscale 값을 깊이(Z축)로 가정하여 Depth Map 생성

- (x, y, depth) 형태의 3D 포인트 클라우드 생성

### (3) MiDaS 기반 Depth Estimation (추가 실험)

- OpenCV 방식의 한계를 보완하기 위해 MiDaS (Monocular Depth Estimation) 적용
  
- 단일 이미지 기반 깊이 추정 모델인 **MiDaS (DPT-Hybrid)** 를 사용

- 단일 RGB 이미지로부터 상대적인 깊이 추정

## 3. 실행 환경
pip install numpy opencv-python pytest torch transformers matplotlib


