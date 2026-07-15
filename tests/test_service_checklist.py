from logic.service_checklist import (
    CHECKLIST_PROFILES,
    ServiceProfile,
    evaluate_transcript,
    get_checklist_profile,
)


def statuses(report):
    return {result.code: result.status for result in report.results}


def test_all_pdf_service_profiles_are_available():
    assert set(CHECKLIST_PROFILES) == {
        "ordinary_point",
        "draft_wall",
        "market",
    }

    assert get_checklist_profile(ServiceProfile.ORDINARY_POINT).title
    assert get_checklist_profile(ServiceProfile.DRAFT_WALL).title
    assert get_checklist_profile(ServiceProfile.MARKET).title


def test_ordinary_point_transcript_marks_core_speech_rules_passed():
    report = evaluate_transcript(
        "Добрый день. Что хотите сегодня попробовать? "
        "К пиву рыбу или чечил добавить? "
        "Нужны сигареты, стики, жвачка, лед, стаканчики? "
        "Итого с вас 4500, чек положу в пакет. "
        "Спасибо за покупку, будем рады видеть вас снова.",
        ServiceProfile.ORDINARY_POINT,
    )

    result_statuses = statuses(report)
    assert result_statuses["greeting"] == "passed"
    assert result_statuses["need_discovery"] == "passed"
    assert result_statuses["snack_offer"] == "passed"
    assert result_statuses["companion_goods_offer"] == "passed"
    assert result_statuses["payment_and_receipt"] == "passed"
    assert result_statuses["farewell_and_business_card"] == "passed"


def test_draft_wall_profile_checks_beer_consultation_and_cashier_handoff():
    report = evaluate_transcript(
        "Здравствуйте. Какое пиво обычно предпочитаете, мягкое или с горчинкой? "
        "Могу дать попробовать этот сорт. К светлому хорошо подойдет рыба или сухарики. "
        "Ваш заказ готов, оплатить можно на кассе. Хорошего вечера.",
        ServiceProfile.DRAFT_WALL,
    )

    result_statuses = statuses(report)
    assert result_statuses["beer_consultation"] == "passed"
    assert result_statuses["snack_offer"] == "passed"
    assert result_statuses["handoff_to_cashier"] == "passed"


def test_market_profile_leaves_non_speech_goods_acceptance_unobserved():
    report = evaluate_transcript(
        "Добрый день. Нужны сигареты, стики, жвачка, лед, стаканчики? "
        "С вас 3200, чек возьмите. Спасибо за покупку, приходите к нам еще.",
        ServiceProfile.MARKET,
    )

    result_statuses = statuses(report)
    assert result_statuses["goods_accepted"] == "unobserved"
    assert result_statuses["companion_goods_offer"] == "passed"


def test_prohibited_speech_fails_even_when_other_phrases_exist():
    report = evaluate_transcript(
        "Здравствуйте. Я не знаю, смотрите сами.",
        ServiceProfile.ORDINARY_POINT,
    )

    assert statuses(report)["no_prohibited_phrases"] == "failed"
