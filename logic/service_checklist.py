from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ServiceProfile(str, Enum):
    ORDINARY_POINT = "ordinary_point"
    DRAFT_WALL = "draft_wall"
    MARKET = "market"
    CALL_CENTER = "call_center"


class EvidenceType(str, Enum):
    SPEECH = "speech"
    VIDEO = "video"
    POS = "pos"
    MANUAL = "manual"


@dataclass(frozen=True)
class ChecklistRule:
    code: str
    title: str
    description: str
    evidence: tuple[EvidenceType, ...]
    speech_patterns: tuple[str, ...] = ()
    negative_patterns: tuple[str, ...] = ()
    required: bool = True
    source: str = ""


@dataclass(frozen=True)
class ChecklistProfile:
    code: ServiceProfile
    title: str
    rules: tuple[ChecklistRule, ...]


@dataclass(frozen=True)
class RuleCheckResult:
    code: str
    title: str
    status: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    missing_reason: str = ""


@dataclass(frozen=True)
class ChecklistReport:
    profile: str
    title: str
    results: tuple[RuleCheckResult, ...]

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.status == "passed")

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if result.status == "failed")

    @property
    def unobserved_count(self) -> int:
        return sum(1 for result in self.results if result.status == "unobserved")

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "title": self.title,
            "summary": {
                "passed": self.passed_count,
                "failed": self.failed_count,
                "unobserved": self.unobserved_count,
                "total": len(self.results),
            },
            "results": [
                {
                    "code": result.code,
                    "title": result.title,
                    "status": result.status,
                    "evidence": list(result.evidence),
                    "missingReason": result.missing_reason,
                }
                for result in self.results
            ],
        }


GREETING_PATTERNS = (
    r"\b(добрый\s+(день|вечер|утро)|здравствуйте|привет|салам(?:алейкум)?)\b",
)

NEED_DISCOVERY_PATTERNS = (
    r"\bчто\s+(хотите|обычно\s+предпочитаете|предпочитаете)\b",
    r"\bподбираете\s+что-то\s+конкретное\b",
    r"\bпомочь\s+с\s+выбором\b",
    r"\bхотите\s+(привычное|что-то\s+новое|попробовать)\b",
    r"\b(светлое|темное|т[её]мное|нефильтрованное)\b",
    r"\b(для\s+праздника|для\s+вечера|на\s+компанию|большой\s+компанией)\b",
)

CONSULTATION_PATTERNS = (
    r"\b(могу|можем|давайте)\s+дать\s+попробовать\b",
    r"\bкакое\s+пиво\s+обычно\s+предпочитаете\b",
    r"\b(мягкое|плотное|с\s+горчинкой)\b",
    r"\b(подсказать|посоветовать)\s+.*\bсорт\b",
)

SNACK_OFFER_PATTERNS = (
    r"\bк\s+(пиву|напитку|светлому|этому)\s+.*\b(рыб[ау]|чечил|снек|орешк|сыр|сухарик|чипс)\b",
    r"\b(рыб[ау]|чечил|снек|орешк|сыр|сухарик|чипс)\s+.*\bдобав(ить|им)\b",
    r"\bзакуск[ау]\s+добавим\b.*\b(рыб[ау]|сыр|снек)\b",
)

COMPANION_GOODS_PATTERNS = (
    r"\b(нужны|положить|не\s+забыли)\s+.*\bсигарет\w*\b.*\bстик\w*\b.*\bжвачк\w*\b.*\bл[её]д\b.*\bстаканчик\w*\b",
)

PAYMENT_PATTERNS = (
    r"\b(сумма|итого|к\s+оплате|с\s+вас)\b",
    r"\b(сдач[аиу]|ваша\s+сдача)\b",
)

CHECK_PATTERNS = (
    r"\b(чек|чека)\b",
)

FAREWELL_PATTERNS = (
    r"\bспасибо\b.*\b(покупк|хорошего|приходите|рады\s+видеть|снова|ещ[её])\b",
    r"\b(благодарим|будем\s+рады\s+видеть|приходите\s+.*ещ[её]|хорошего\s+(дня|вечера))\b",
)

DRAFT_HANDOFF_PATTERNS = (
    r"\b(заказ\s+готов|готово|передаю\s+заказ)\b",
    r"\b(оплатить|пройдите|подойдите|проходите)\s+.*\b(касс[аеуы]|на\s+кассу)\b",
)

PROHIBITED_SPEECH_PATTERNS = (
    r"\b(я\s+не\s+знаю|смотрите\s+сами|это\s+не\s+ко\s+мне|вс[её]\s+там|нету)\b",
    r"\b(заткнись|дурак|иди\s+отсюда)\b",
)


# ===========================
# CALL CENTER PATTERNS
# ===========================

CALL_CENTER_GREETING_PATTERNS = (
    r"\b(добрый\s+(день|вечер|утро)|здравствуйте|приветствую|компания\s+\w+)\b",
    r"\b(слушаю|чем\s+могу\s+помочь|как\s+я\s+могу\s+помочь)\b",
)

CALL_CENTER_INTRO_PATTERNS = (
    r"\b(меня\s+зовут|это\s+\w+\s+из\s+компании|представляю\s+компанию)\b",
    r"\b(звоню\s+по\s+поводу|обращаюсь\s+к\s+вам)\b",
)

CALL_CENTER_NEED_DISCOVERY_PATTERNS = (
    r"\b(расскажите|опишите|в\s+ч[её]м\s+проблема|что\s+случилось)\b",
    r"\b(как\s+я\s+могу\s+помочь|чем\s+могу\s+быть\s+полезен)\b",
)

CALL_CENTER_SOLUTION_PATTERNS = (
    r"\b(предлагаю|решение|вариант|давайте\s+сделаем|можем\s+предложить)\b",
    r"\b(оформим|подключим|отправим|проверим|уточним)\b",
)

CALL_CENTER_CONFIRMATION_PATTERNS = (
    r"\b(вс[её]\s+верно|правильно\s+я\s+понимаю|уточню\s+ещ[её]\s+раз)\b",
    r"\b(повторите|продиктуйте|назовите)\b",
)

CALL_CENTER_CLOSING_PATTERNS = (
    r"\b(всего\s+доброго|хорошего\s+дня|до\s+свидания|будем\s+на\s+связи)\b",
    r"\b(обращайтесь|звоните|если\s+будут\s+вопросы)\b",
)

CALL_CENTER_EMPATHY_PATTERNS = (
    r"\b(понимаю|сожалею|приношу\s+извинения|извините\s+за)\b",
    r"\b(неудобства|неприятности|задержку|ожидание)\b",
)


COMMON_RULES = (
    ChecklistRule(
        code="greeting",
        title="Приветствие",
        description="Сотрудник первым приветствует покупателя вежливо и обращенно к нему.",
        evidence=(EvidenceType.SPEECH,),
        speech_patterns=GREETING_PATTERNS,
        source="Регламент п.5; приложения 1-3 п.1",
    ),
    ChecklistRule(
        code="no_prohibited_phrases",
        title="Нет запрещенных фраз",
        description="Сотрудник не использует грубые или безразличные фразы.",
        evidence=(EvidenceType.SPEECH,),
        negative_patterns=PROHIBITED_SPEECH_PATTERNS,
        source="Регламент п.4, п.17",
    ),
)


ORDINARY_PROFILE = ChecklistProfile(
    code=ServiceProfile.ORDINARY_POINT,
    title="Бармен-кассир обычной точки",
    rules=COMMON_RULES + (
        ChecklistRule(
            code="need_discovery",
            title="Выявление потребности",
            description="Сотрудник задает открытый или уточняющий вопрос либо применяет активное слушание.",
            evidence=(EvidenceType.SPEECH,),
            speech_patterns=NEED_DISCOVERY_PATTERNS,
            source="Приложение 1 п.2; регламент п.6-8",
        ),
        ChecklistRule(
            code="snack_offer",
            title="Предложение снеков/закусок",
            description="Предложение конкретное, с выбором или подходящим вариантом товара.",
            evidence=(EvidenceType.SPEECH,),
            speech_patterns=SNACK_OFFER_PATTERNS,
            source="Приложение 1 п.3; регламент п.10",
        ),
        ChecklistRule(
            code="companion_goods_offer",
            title="Предложение сопутствующих товаров",
            description="Произнесен скрипт про сигареты, стики, жвачку, лед и стаканчики.",
            evidence=(EvidenceType.SPEECH,),
            speech_patterns=COMPANION_GOODS_PATTERNS,
            source="Приложение 1 п.4; регламент п.11",
        ),
        ChecklistRule(
            code="payment_and_receipt",
            title="Оплата и чек",
            description="Сотрудник озвучивает оплату/сдачу и выдает или предлагает чек.",
            evidence=(EvidenceType.SPEECH, EvidenceType.POS),
            speech_patterns=PAYMENT_PATTERNS + CHECK_PATTERNS,
            source="Приложение 1 п.5; регламент п.13",
        ),
        ChecklistRule(
            code="farewell_and_business_card",
            title="Прощание и визитка",
            description="Сотрудник благодарит, приглашает вернуться и передает визитку.",
            evidence=(EvidenceType.SPEECH, EvidenceType.VIDEO),
            speech_patterns=FAREWELL_PATTERNS,
            source="Приложение 1 п.6; регламент п.14",
        ),
    ),
)


DRAFT_WALL_PROFILE = ChecklistProfile(
    code=ServiceProfile.DRAFT_WALL,
    title="Бармен-универсал за разливной стенкой",
    rules=COMMON_RULES + (
        ChecklistRule(
            code="need_discovery",
            title="Выявление потребности по пиву",
            description="Сотрудник уточняет сорт, объем, вкус или задачу покупки.",
            evidence=(EvidenceType.SPEECH,),
            speech_patterns=NEED_DISCOVERY_PATTERNS,
            source="Приложение 2 п.2; регламент п.6-8",
        ),
        ChecklistRule(
            code="beer_consultation",
            title="Консультация и помощь с сортом",
            description="Сотрудник помогает выбрать сорт и предлагает дегустацию при сомнении.",
            evidence=(EvidenceType.SPEECH,),
            speech_patterns=CONSULTATION_PATTERNS,
            source="Приложение 2 п.3; регламент п.8-9",
        ),
        ChecklistRule(
            code="snack_offer",
            title="Предложение снеков/закусок",
            description="Предложение конкретное и связано с выбранным пивом.",
            evidence=(EvidenceType.SPEECH,),
            speech_patterns=SNACK_OFFER_PATTERNS,
            source="Приложение 2 п.4; регламент п.10",
        ),
        ChecklistRule(
            code="handoff_to_cashier",
            title="Передача заказа и направление на кассу",
            description="Сотрудник передает заказ, при необходимости озвучивает сорт/объем и направляет на оплату.",
            evidence=(EvidenceType.SPEECH, EvidenceType.VIDEO),
            speech_patterns=DRAFT_HANDOFF_PATTERNS,
            source="Приложение 2 п.5",
        ),
        ChecklistRule(
            code="farewell",
            title="Вежливое завершение",
            description="Сотрудник благодарит или прощается с покупателем.",
            evidence=(EvidenceType.SPEECH,),
            speech_patterns=FAREWELL_PATTERNS,
            source="Приложение 2 п.5; регламент п.14",
        ),
    ),
)


MARKET_PROFILE = ChecklistProfile(
    code=ServiceProfile.MARKET,
    title="Бармен-кассир/кассир точки Маркета",
    rules=COMMON_RULES + (
        ChecklistRule(
            code="goods_accepted",
            title="Прием товара к оплате",
            description="Кассир принимает товар, пробивает позиции и при необходимости уточняет количество или наименование.",
            evidence=(EvidenceType.VIDEO, EvidenceType.POS, EvidenceType.MANUAL),
            source="Приложение 3 п.2",
        ),
        ChecklistRule(
            code="companion_goods_offer",
            title="Предложение сопутствующих товаров",
            description="Произнесен скрипт про сигареты, стики, жвачку, лед и стаканчики.",
            evidence=(EvidenceType.SPEECH,),
            speech_patterns=COMPANION_GOODS_PATTERNS,
            source="Приложение 3 п.3; регламент п.11",
        ),
        ChecklistRule(
            code="payment_and_receipt",
            title="Оплата и чек",
            description="Сотрудник озвучивает оплату/сдачу и выдает или предлагает чек.",
            evidence=(EvidenceType.SPEECH, EvidenceType.POS),
            speech_patterns=PAYMENT_PATTERNS + CHECK_PATTERNS,
            source="Приложение 3 п.4; регламент п.13",
        ),
        ChecklistRule(
            code="farewell_and_business_card",
            title="Прощание и визитка",
            description="Сотрудник благодарит, приглашает вернуться и передает визитку.",
            evidence=(EvidenceType.SPEECH, EvidenceType.VIDEO),
            speech_patterns=FAREWELL_PATTERNS,
            source="Приложение 3 п.5; регламент п.14",
        ),
    ),
)


CALL_CENTER_RULES = (
    ChecklistRule(
        code="greeting",
        title="Приветствие и представление",
        description="Оператор приветствует клиента и представляется.",
        evidence=(EvidenceType.SPEECH,),
        speech_patterns=CALL_CENTER_GREETING_PATTERNS + CALL_CENTER_INTRO_PATTERNS,
        source="Стандарт обслуживания п.1",
    ),
    ChecklistRule(
        code="no_prohibited_phrases",
        title="Нет запрещенных фраз",
        description="Оператор не использует грубые или безразличные фразы.",
        evidence=(EvidenceType.SPEECH,),
        negative_patterns=PROHIBITED_SPEECH_PATTERNS,
        source="Стандарт обслуживания п.2",
    ),
    ChecklistRule(
        code="need_discovery",
        title="Выявление потребности",
        description="Оператор задает уточняющие вопросы для понимания проблемы.",
        evidence=(EvidenceType.SPEECH,),
        speech_patterns=CALL_CENTER_NEED_DISCOVERY_PATTERNS,
        source="Стандарт обслуживания п.3",
    ),
    ChecklistRule(
        code="empathy",
        title="Эмпатия и работа с возражениями",
        description="Оператор проявляет понимание и эмпатию к ситуации клиента.",
        evidence=(EvidenceType.SPEECH,),
        speech_patterns=CALL_CENTER_EMPATHY_PATTERNS,
        source="Стандарт обслуживания п.4",
    ),
    ChecklistRule(
        code="solution_offer",
        title="Предложение решения",
        description="Оператор предлагает конкретное решение или варианты действий.",
        evidence=(EvidenceType.SPEECH,),
        speech_patterns=CALL_CENTER_SOLUTION_PATTERNS,
        source="Стандарт обслуживания п.5",
    ),
    ChecklistRule(
        code="confirmation",
        title="Подтверждение и уточнение",
        description="Оператор подтверждает понимание и уточняет детали.",
        evidence=(EvidenceType.SPEECH,),
        speech_patterns=CALL_CENTER_CONFIRMATION_PATTERNS,
        source="Стандарт обслуживания п.6",
    ),
    ChecklistRule(
        code="closing",
        title="Завершение разговора",
        description="Оператор вежливо завершает разговор и приглашает обращаться снова.",
        evidence=(EvidenceType.SPEECH,),
        speech_patterns=CALL_CENTER_CLOSING_PATTERNS,
        source="Стандарт обслуживания п.7",
    ),
)


CALL_CENTER_PROFILE = ChecklistProfile(
    code=ServiceProfile.CALL_CENTER,
    title="Оператор колл-центра",
    rules=CALL_CENTER_RULES,
)


CHECKLIST_PROFILES = {
    profile.code.value: profile
    for profile in (ORDINARY_PROFILE, DRAFT_WALL_PROFILE, MARKET_PROFILE, CALL_CENTER_PROFILE)
}


def get_checklist_profile(profile: str | ServiceProfile) -> ChecklistProfile:
    key = profile.value if isinstance(profile, ServiceProfile) else profile
    try:
        return CHECKLIST_PROFILES[key]
    except KeyError as error:
        valid = ", ".join(sorted(CHECKLIST_PROFILES))
        raise ValueError(f"Unknown service checklist profile {key!r}. Valid: {valid}") from error


def evaluate_transcript(
    text: str,
    profile: str | ServiceProfile = ServiceProfile.ORDINARY_POINT,
) -> ChecklistReport:
    checklist = get_checklist_profile(profile)
    normalized_text = _normalize_text(text)
    results = []

    for rule in checklist.rules:
        negative_matches = _matching_patterns(normalized_text, rule.negative_patterns)
        positive_matches = _matching_patterns(normalized_text, rule.speech_patterns)

        if negative_matches:
            results.append(RuleCheckResult(
                code=rule.code,
                title=rule.title,
                status="failed",
                evidence=negative_matches,
                missing_reason="Detected prohibited speech.",
            ))
        elif rule.negative_patterns and not rule.speech_patterns:
            results.append(RuleCheckResult(
                code=rule.code,
                title=rule.title,
                status="passed",
            ))
        elif positive_matches:
            results.append(RuleCheckResult(
                code=rule.code,
                title=rule.title,
                status="passed",
                evidence=positive_matches,
            ))
        elif EvidenceType.SPEECH in rule.evidence and rule.speech_patterns:
            results.append(RuleCheckResult(
                code=rule.code,
                title=rule.title,
                status="failed",
                missing_reason="Required speech evidence was not found in transcript.",
            ))
        else:
            results.append(RuleCheckResult(
                code=rule.code,
                title=rule.title,
                status="unobserved",
                missing_reason="Rule requires video, POS, or manual evidence.",
            ))

    return ChecklistReport(
        profile=checklist.code.value,
        title=checklist.title,
        results=tuple(results),
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _matching_patterns(text: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
    if not text or not patterns:
        return ()
    return tuple(pattern for pattern in patterns if re.search(pattern, text))
