from audio.rtsp_transcriber import RTSPVisitTranscriber


def test_known_whisper_hallucinations_are_removed():
    clean = RTSPVisitTranscriber._clean_text
    assert clean("Продолжение следует...") == ""
    assert clean("Спасибо за просмотр!") == ""
    assert clean("Субтитры создавал DimaTorzhok") == ""
    assert clean("Автор субтитров — DimaTorzhok") == ""
    assert clean("Пакет нужен?") == "Пакет нужен?"
