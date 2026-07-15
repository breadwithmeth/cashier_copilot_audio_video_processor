from vision.person_detector import PersonDetector


def test_hand_positions():
    classify = PersonDetector._classify_hand
    assert classify((100, 100), (100, 80), (100, 60)) == "raised"
    assert classify((100, 100), (140, 110), (180, 120)) == "extended"
    assert classify((100, 100), (100, 130), (100, 160)) == "down"
    assert classify((100, 100), (100, 140), (100, 120)) == "bent"
    assert classify(None, None, (100, 100)) == "unknown"


def test_person_bbox_is_limited_to_upper_body():
    assert PersonDetector._upper_body_bbox((10, 20, 110, 220)) == (
        10,
        20,
        110,
        130,
    )


def test_person_role_uses_upper_body_center():
    detector = PersonDetector.__new__(PersonDetector)
    detector.customer_roi = (0, 0, 100, 100)
    detector.cashier_roi = (0, 120, 100, 220)

    assert detector._roles_for_bbox((10, 20, 90, 80)) == ["customer"]
    assert detector._roles_for_bbox((10, 140, 90, 200)) == ["cashier"]
