from judex.matcher import BNSMatcher


def best_section(text: str) -> int:
    return BNSMatcher().rank(text, limit=1)[0].section.number


def test_unseen_theft_language() -> None:
    assert best_section("My wallet disappeared from my backpack and I believe somebody took it") == 303


def test_snatching_beats_general_theft() -> None:
    assert best_section("A stranger grabbed my phone from my hand and ran away") == 304


def test_personation_cheating_is_distinguished_from_general_cheating() -> None:
    assert best_section("A fake bank officer called me and convinced me to transfer money") == 319


def test_ransom_is_ranked_as_kidnapping_for_ransom() -> None:
    assert best_section("They kidnapped him and demanded money to release him") == 140


def test_house_breaking_beats_general_theft() -> None:
    assert best_section("Someone broke the lock and entered my house before taking jewellery") == 330


def test_fracture_prefers_grievous_hurt() -> None:
    assert best_section("He hit me and broke my arm") in {116, 117}


def test_criminal_intimidation() -> None:
    assert best_section("He threatened to harm me if I reported him") == 351
