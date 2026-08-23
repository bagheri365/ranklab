from ranklab.analysis.contrasts import pairwise_margin, regime_contrast, target_contrast


def test_direct_contrasts() -> None:
    standard = pairwise_margin(0.42, 0.40)
    randomized = pairwise_margin(0.38, 0.41)
    assert regime_contrast(standard, randomized) == standard - randomized
    assert target_contrast(0.03, -0.01) == 0.04
