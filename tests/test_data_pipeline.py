import numpy as np
from data.argo_preprocess import pressure_to_depth, TARGET_DEPTHS


def test_pressure_to_depth_reasonable():
    # Whether gsw (TEOS-10) is installed or not, pressure->depth should
    # stay within ~2% of the 1 dbar ~= 1 m rule of thumb -- exact only
    # without gsw, approximate but still close with it.
    p = np.array([0, 10, 50, 100, 500])
    d = pressure_to_depth(p, 10.0)
    assert np.allclose(d, p, rtol=0.02, atol=1.0)


def test_target_depths_present():
    assert TARGET_DEPTHS[0] == 0
    assert TARGET_DEPTHS[-1] == 1000


def test_predict_point_works_globally_not_just_arabian_sea():
    """Regression test for the map-click domain fix: /api/predict-point
    must answer for ocean points far outside the training dataset's
    Arabian Sea extent (falling back to extrapolation when live fetch
    isn't available) instead of hard-rejecting them, and must accept
    negative lat/lon in the URL (Southern/Western Hemisphere)."""
    import api_server as srv
    client = srv.app.test_client()

    # Mid-Pacific: far outside the Arabian Sea training box, negative lon.
    r = client.get("/api/predict-point/0.0/-160.0")
    assert r.status_code == 200
    body = r.get_json()
    assert body["interpolation"]["source"] in ("validated_dataset", "live_satellite", "extrapolated")
    assert "predicted" in body

    # A land point should still be rejected (no ocean profile exists there).
    r_land = client.get("/api/predict-point/28.6/77.2")  # New Delhi
    assert r_land.status_code == 400
    assert "land" in r_land.get_json()["error"].lower()
