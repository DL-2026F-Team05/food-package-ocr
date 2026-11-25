"""
AI 1 OCR 모듈 — 식품 패키지 사진에서 글자 줄을 읽어 JSON 형태로 돌려줍니다.

모델: PaddleOCR PP-OCRv5 (검출: PP-OCRv5_mobile_det / 인식: korean_PP-OCRv5_mobile_rec, 한국어+영어+숫자)
설정: 3주차 실험에서 가장 좋았던 기준 설정(A_기준) — 전체 CER 27.1% (핵심 정보 줄 기준)

필요 패키지 (CPU):
    pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    pip install paddleocr==3.7.0

사용법:
    from ocr_engine import OCREngine
    engine = OCREngine()                 # 처음 한 번만 (모델 불러오기, 몇 초 걸림)
    result = engine.run('P004_1.jpg')    # 파일 경로 / bytes / PIL.Image / OpenCV(numpy, BGR) 모두 가능
    print(result['full_text'])

명령줄:
    python ocr_engine.py 사진.jpg        # 결과 JSON 출력

출력 형식:
    {
      "image": "P004_1.jpg",
      "width": 3000, "height": 4000,
      "lines": [
        {"text": "2026.12.15 까지", "box": [x1, y1, x2, y2],
         "poly": [[x, y], [x, y], [x, y], [x, y]], "confidence": 0.97},
        ...
      ],
      "full_text": "줄1\n줄2\n..."
    }
    - box/poly 좌표는 사진의 방향을 바로잡은(EXIF 회전 적용) 이미지 기준 픽셀 좌표입니다.
    - confidence는 0~1, 낮을수록 잘못 읽었을 가능성이 큽니다.
"""
import io
import os
import json

import numpy as np
from PIL import Image, ImageOps


class OCREngine:
    def __init__(self, device='cpu', min_confidence=0.0):
        """device: 'cpu' 또는 'gpu:0' / min_confidence: 이 값보다 신뢰도가 낮은 줄은 버림"""
        from paddleocr import PaddleOCR
        self.min_confidence = min_confidence
        self.ocr = PaddleOCR(
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='korean_PP-OCRv5_mobile_rec',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=1600,
            text_det_limit_type='max',
            device=device,
        )

    @staticmethod
    def _to_bgr(image):
        """여러 입력 형태를 PaddleOCR이 받는 BGR numpy 배열로 바꿉니다."""
        name = None
        if isinstance(image, np.ndarray):          # OpenCV 이미지 (BGR 또는 흑백)
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            return np.ascontiguousarray(arr), name
        if isinstance(image, (str, os.PathLike)):  # 파일 경로
            name = os.path.basename(str(image))
            pil = Image.open(image)
        elif isinstance(image, (bytes, bytearray)):  # 업로드된 파일 내용
            pil = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            pil = image
        else:
            raise TypeError(f'지원하지 않는 입력 형태입니다: {type(image)}')
        pil = ImageOps.exif_transpose(pil).convert('RGB')   # 폰 사진 방향 바로잡기
        return np.ascontiguousarray(np.array(pil)[:, :, ::-1]), name

    def run(self, image, image_name=None):
        arr, name = self._to_bgr(image)
        res = self.ocr.predict(arr)[0]
        lines = []
        for text, score, poly in zip(res['rec_texts'], res['rec_scores'], res['rec_polys']):
            score = float(score)
            if not str(text).strip() or score < self.min_confidence:
                continue
            p = np.asarray(poly).astype(int)
            lines.append({
                'text': str(text),
                'box': [int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())],
                'poly': p.tolist(),
                'confidence': round(score, 4),
            })
        return {
            'image': image_name or name or '',
            'width': int(arr.shape[1]),
            'height': int(arr.shape[0]),
            'lines': lines,
            'full_text': '\n'.join(l['text'] for l in lines),
        }

    def run_to_json(self, image, out_path, image_name=None):
        result = self.run(image, image_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('사용법: python ocr_engine.py 사진.jpg [사진2.jpg ...]')
        sys.exit(1)
    engine = OCREngine()
    for path in sys.argv[1:]:
        print(json.dumps(engine.run(path), ensure_ascii=False, indent=2))
