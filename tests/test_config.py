from sdpo import config


def test_constants_match_spec():
    assert config.N_UNIVERSE * config.MAX_WEIGHT >= 1  # cap is feasible
    assert list(config.TEST_YEARS) == [2016, 2017, 2018, 2019, 2020]
    assert len(config.FEATURES) == 7 and config.SEQ_LEN == 12
    assert (config.BL_DELTA, config.BL_TAU, config.COST_BPS) == (2.5, 0.05, 10)
    assert config.ARTIFACTS_DIR.name == "artifacts"
