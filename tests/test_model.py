import numpy as np
from model import preprocess_img


def test_preprocess_img():
    img_path = 'static/Images/seg_pred/28.jpg'
    img_processed = preprocess_img(img_path)
    assert np.min(img_processed) >= 0
    assert np.max(img_processed) <= 1
