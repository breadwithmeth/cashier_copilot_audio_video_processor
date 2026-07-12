from vision.person_detector import PersonDetector


def test_hand_positions():
    classify = PersonDetector._classify_hand
    assert classify((100, 100), (100, 80), (100, 60)) == "raised"
    assert classify((100, 100), (140, 110), (180, 120)) == "extended"
    assert classify((100, 100), (100, 130), (100, 160)) == "down"
    assert classify((100, 100), (100, 140), (100, 120)) == "bent"
    assert classify(None, None, (100, 100)) == "unknown"
