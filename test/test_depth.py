import numpy as np
import pytest
from depth import generate_depth_map

def test_generate_depth_map_shape_and_type():
    image = np.zeros((100, 100, 3), dtype=np.uint8)  # 검정색 빈 이미지
    depth_map = generate_depth_map(image)

    assert depth_map.shape == image.shape, "출력 크기가 입력 크기와 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."

def test_generate_depth_map_raises_on_none():
    with pytest.raises(ValueError):
        generate_depth_map(None)
