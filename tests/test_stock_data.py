from stock_data import build_insight_bullets


def test_build_insight_bullets_highlights_growth_and_momentum():
    profile = {
        "revenue_growth": 24.8,
        "profit_margin": 22.1,
        "change_pct": 3.4,
        "pe": 44.2,
        "market_status": "Open",
    }

    bullets = build_insight_bullets(profile)

    assert len(bullets) >= 3
    assert any("growth" in bullet.lower() for bullet in bullets)
    assert any("momentum" in bullet.lower() for bullet in bullets)
