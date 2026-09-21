from src.metrics import mae, rms_wape, wape


def test_wape_and_rms():
    y = [100.0, 200.0, 300.0]
    p = [110.0, 180.0, 330.0]
    assert abs(wape(y, p) - (10 + 20 + 30) / 600 * 100) < 1e-9
    assert abs(mae(y, p) - 20.0) < 1e-9
    assert abs(rms_wape([10.0, 20.0]) - (0.5 * (100 + 400)) ** 0.5) < 1e-9
