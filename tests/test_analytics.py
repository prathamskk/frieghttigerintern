import pytest
import pandas as pd
import numpy as np
from src.analytics import (
    load_and_preprocess_shipments,
    compute_weekly_route_aggregates,
    compute_baselines_and_flags
)

# ----------------------------------------------------------------------
# 1. TEST WEEK GROUPING (Monday-Sunday Cohorting)
# ----------------------------------------------------------------------
def test_week_grouping_all_days_of_week():
    """
    Verify that all shipment_date values are grouped into Monday-to-Sunday
    calendar weeks where week_of strictly represents that week's Monday date.
    """
    # Sample spanning an entire week: Monday 2024-01-01 to Sunday 2024-01-07
    # and next Monday 2024-01-08
    test_dates = [
        ("2024-01-01", "Monday", "2024-01-01"),
        ("2024-01-02", "Tuesday", "2024-01-01"),
        ("2024-01-03", "Wednesday", "2024-01-01"),
        ("2024-01-04", "Thursday", "2024-01-01"),
        ("2024-01-05", "Friday", "2024-01-01"),
        ("2024-01-06", "Saturday", "2024-01-01"),
        ("2024-01-07", "Sunday", "2024-01-01"),
        ("2024-01-08", "Next Monday", "2024-01-08"),
        ("2024-01-14", "Next Sunday", "2024-01-08"),
    ]

    records = []
    for i, (dt, day_name, expected_monday) in enumerate(test_dates):
        records.append({
            "shipment_id": f"TEST_{i}",
            "origin": "Mumbai",
            "destination": "Pune",
            "route_type": "Short",
            "material": "Cement",
            "quantity_tonnes": 10.0,
            "distance_km": 150.0,
            "freight_cost_inr": 5000,
            "shipment_date": dt,
            "transporter": "Test Logistics"
        })

    temp_csv = "tests_temp_dates.csv"
    pd.DataFrame(records).to_csv(temp_csv, index=False)

    try:
        processed = load_and_preprocess_shipments(temp_csv)
        for i, (dt, day_name, expected_monday) in enumerate(test_dates):
            actual_week_of = processed.loc[i, "week_of"]
            assert actual_week_of == expected_monday, (
                f"Date {dt} ({day_name}) got week_of {actual_week_of}, expected {expected_monday}"
            )
            # Verify the parsed date is strictly a Monday
            monday_dt = pd.to_datetime(actual_week_of)
            assert monday_dt.weekday() == 0, f"week_of {actual_week_of} is not a Monday!"
    finally:
        import os
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

def test_week_grouping_on_dataset():
    """
    Verify across the full shipment_records.csv that every week_of is strictly a Monday.
    """
    df = load_and_preprocess_shipments("casestudy/shipment_records.csv")
    assert len(df) == 2940

    # Ensure week_of is strictly Monday (weekday == 0)
    mondays = pd.to_datetime(df['week_of'])
    assert (mondays.dt.weekday == 0).all(), "Found non-Monday dates in week_of!"

    # Ensure difference between shipment_date and week_of is between 0 and 6 days
    diff_days = (df['shipment_date'] - mondays).dt.days
    assert (diff_days >= 0).all(), "Found shipment_date before week_of Monday!"
    assert (diff_days <= 6).all(), "Found shipment_date more than 6 days after week_of Monday!"


# ----------------------------------------------------------------------
# 2. TEST METRIC FORMULA: cost_per_tonne_km
# ----------------------------------------------------------------------
def test_metric_formula_calculation():
    """
    Confirm that cost_per_tonne_km is computed exactly as:
    total freight cost / (quantity in tonnes * distance in km)
    """
    # Two shipments on the same route in the same week
    # Shipment 1: 10 tonnes, 100 km, 5,000 INR -> tonne_km = 1,000
    # Shipment 2: 20 tonnes, 200 km, 12,000 INR -> tonne_km = 4,000
    # Total cost = 17,000 INR
    # Total tonne-km = 1,000 + 4,000 = 5,000 tonne-km
    # Expected weighted cost_per_tonne_km = 17,000 / 5,000 = 3.40 INR / tonne-km
    sample_data = pd.DataFrame([
        {
            "shipment_id": "S1", "origin": "Mumbai", "destination": "Pune", "route_type": "Short",
            "quantity_tonnes": 10.0, "distance_km": 100.0, "freight_cost_inr": 5000,
            "shipment_date": pd.to_datetime("2024-01-02"), "week_of": "2024-01-01",
            "tonne_km": 1000.0, "route": "Mumbai-Pune"
        },
        {
            "shipment_id": "S2", "origin": "Mumbai", "destination": "Pune", "route_type": "Short",
            "quantity_tonnes": 20.0, "distance_km": 200.0, "freight_cost_inr": 12000,
            "shipment_date": pd.to_datetime("2024-01-04"), "week_of": "2024-01-01",
            "tonne_km": 4000.0, "route": "Mumbai-Pune"
        }
    ])

    weekly = compute_weekly_route_aggregates(sample_data)
    assert len(weekly) == 1
    actual_cost_per_tkm = weekly.iloc[0]['cost_per_tonne_km']
    expected_cost_per_tkm = 17000.0 / 5000.0  # 3.40
    assert np.isclose(actual_cost_per_tkm, expected_cost_per_tkm, atol=1e-6), (
        f"Expected {expected_cost_per_tkm}, got {actual_cost_per_tkm}"
    )

def test_metric_formula_dataset_benchmarks():
    """
    Verify metric formula matches exact values in official sample_output_format_v2.csv:
    - Delhi-Jaipur (2024-11-11): 4.17
    - Ahmedabad-Mumbai (2025-01-20): 3.29
    - Mumbai-Pune (2025-09-15): 3.98
    """
    df = load_and_preprocess_shipments("casestudy/shipment_records.csv")
    weekly = compute_weekly_route_aggregates(df)

    benchmarks = [
        ("Delhi-Jaipur", "2024-11-11", 4.17),
        ("Ahmedabad-Mumbai", "2025-01-20", 3.29),
        ("Mumbai-Pune", "2025-09-15", 3.98)
    ]

    for route, week_of, expected_val in benchmarks:
        row = weekly[(weekly['route'] == route) & (weekly['week_of'] == week_of)]
        assert not row.empty, f"Missing route-week: {route} {week_of}"
        actual_val = round(row.iloc[0]['cost_per_tonne_km'], 2)
        assert actual_val == expected_val, (
            f"Benchmark mismatch for {route} on {week_of}: Expected {expected_val}, got {actual_val}"
        )


# ----------------------------------------------------------------------
# 3. TEST ROUTE GROUPING: Exact origin-destination & route_type bucket
# ----------------------------------------------------------------------
def test_route_and_route_type_grouping():
    """
    Confirm records are grouped by exact route (same origin and destination)
    and the provided route_type bucket without altering buckets.
    """
    sample_data = pd.DataFrame([
        # Route 1: Mumbai-Pune (Short)
        {"shipment_id": "S1", "origin": "Mumbai", "destination": "Pune", "route_type": "Short",
         "quantity_tonnes": 10.0, "distance_km": 150.0, "freight_cost_inr": 6000,
         "shipment_date": pd.to_datetime("2024-01-02"), "week_of": "2024-01-01",
         "tonne_km": 1500.0, "route": "Mumbai-Pune"},
        # Route 2: Mumbai-Delhi (Long) - same origin, different destination!
        {"shipment_id": "S2", "origin": "Mumbai", "destination": "Delhi", "route_type": "Long",
         "quantity_tonnes": 10.0, "distance_km": 1400.0, "freight_cost_inr": 35000,
         "shipment_date": pd.to_datetime("2024-01-03"), "week_of": "2024-01-01",
         "tonne_km": 14000.0, "route": "Mumbai-Delhi"},
        # Route 3: Pune-Mumbai (Short) - reverse direction!
        {"shipment_id": "S3", "origin": "Pune", "destination": "Mumbai", "route_type": "Short",
         "quantity_tonnes": 10.0, "distance_km": 150.0, "freight_cost_inr": 6200,
         "shipment_date": pd.to_datetime("2024-01-04"), "week_of": "2024-01-01",
         "tonne_km": 1500.0, "route": "Pune-Mumbai"}
    ])

    weekly = compute_weekly_route_aggregates(sample_data)
    # Must produce 3 distinct groups
    assert len(weekly) == 3
    routes = set(weekly['route'].tolist())
    assert routes == {"Mumbai-Pune", "Mumbai-Delhi", "Pune-Mumbai"}

    # Confirm route_types are strictly preserved
    mp_type = weekly[weekly['route'] == 'Mumbai-Pune'].iloc[0]['route_type']
    md_type = weekly[weekly['route'] == 'Mumbai-Delhi'].iloc[0]['route_type']
    pm_type = weekly[weekly['route'] == 'Pune-Mumbai'].iloc[0]['route_type']
    assert mp_type == "Short"
    assert md_type == "Long"
    assert pm_type == "Short"


# ----------------------------------------------------------------------
# 4. TEST TRAILING 8-WEEK WINDOW (Zero Look-Ahead Bias)
# ----------------------------------------------------------------------
def test_trailing_8_week_window_strictly_prior_weeks():
    """
    Confirm the rolling average uses strictly prior weeks (t-8 to t-1)
    with zero look-ahead bias (the current week must not be included).
    """
    # Create 12 sequential weeks with known costs 1.0, 2.0, ..., 12.0
    weeks = [f"2024-01-{i:02d}" for i in range(1, 13)]
    rows = []
    for i, w in enumerate(weeks, start=1):
        rows.append({
            "route": "Test-Route",
            "origin": "Test",
            "destination": "Route",
            "route_type": "Short",
            "week_of": w,
            "total_cost": float(i) * 1000.0,
            "total_tonne_km": 1000.0,
            "cost_per_tonne_km": float(i),  # Cost = 1.0, 2.0, ..., 12.0
            "shipment_count": 5
        })
    weekly = pd.DataFrame(rows)

    all_metrics, _ = compute_baselines_and_flags(weekly, spike_threshold_pct=20.0)

    # Week 9 (idx = 8):
    # Strictly prior 8 weeks: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    # Expected mean = sum(1..8)/8 = 36/8 = 4.5
    # Current week cost is 9.0; must NOT be in the average!
    w9_row = all_metrics[all_metrics['week_of'] == weeks[8]].iloc[0]
    assert np.isclose(w9_row['hist_avg'], 4.5), f"Expected 4.5, got {w9_row['hist_avg']}"
    assert w9_row['prior_weeks_used'] == 8

    # Week 10 (idx = 9):
    # Strictly prior 8 weeks: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    # Expected mean = sum(2..9)/8 = 44/8 = 5.5
    # Current week cost is 10.0; must NOT be in the average!
    w10_row = all_metrics[all_metrics['week_of'] == weeks[9]].iloc[0]
    assert np.isclose(w10_row['hist_avg'], 5.5), f"Expected 5.5, got {w10_row['hist_avg']}"
    assert w10_row['prior_weeks_used'] == 8


# ----------------------------------------------------------------------
# 5. TEST SHORT HISTORY EDGE CASE (< 8 Prior Weeks Available)
# ----------------------------------------------------------------------
def test_short_history_no_padding_or_extrapolation():
    """
    If a route has fewer than 8 prior weeks, confirm it uses all available
    prior weeks without padding or extrapolating.
    """
    weeks = [f"2024-01-{i:02d}" for i in range(1, 6)]
    rows = []
    # 5 weeks: costs 2.0, 4.0, 6.0, 8.0, 10.0
    for i, w in enumerate(weeks, start=1):
        rows.append({
            "route": "Short-History-Route",
            "origin": "Short",
            "destination": "Route",
            "route_type": "Medium",
            "week_of": w,
            "total_cost": float(i * 2) * 1000.0,
            "total_tonne_km": 1000.0,
            "cost_per_tonne_km": float(i * 2),
            "shipment_count": 4
        })
    weekly = pd.DataFrame(rows)

    all_metrics, _ = compute_baselines_and_flags(weekly, spike_threshold_pct=20.0)

    # Week 1: 0 prior weeks
    w1 = all_metrics.iloc[0]
    assert np.isnan(w1['hist_avg']), "Week 1 should have NaN hist_avg (no prior history)"
    assert w1['prior_weeks_used'] == 0
    assert w1['vs_own_history'] == "No prior history available"

    # Week 2: 1 prior week (cost 2.0)
    w2 = all_metrics.iloc[1]
    assert np.isclose(w2['hist_avg'], 2.0), f"Expected 2.0, got {w2['hist_avg']}"
    assert w2['prior_weeks_used'] == 1

    # Week 3: 2 prior weeks (costs 2.0, 4.0) -> mean = 3.0
    w3 = all_metrics.iloc[2]
    assert np.isclose(w3['hist_avg'], 3.0), f"Expected 3.0, got {w3['hist_avg']}"
    assert w3['prior_weeks_used'] == 2

    # Week 5: 4 prior weeks (costs 2.0, 4.0, 6.0, 8.0) -> mean = 5.0
    w5 = all_metrics.iloc[4]
    assert np.isclose(w5['hist_avg'], 5.0), f"Expected 5.0, got {w5['hist_avg']}"
    assert w5['prior_weeks_used'] == 4


# ----------------------------------------------------------------------
# 6. TEST PEER BASELINE: Excludes Target Route & Filters by route_type
# ----------------------------------------------------------------------
def test_peer_baseline_excludes_self_and_filters_by_route_type():
    """
    Ensure the peer average for the same week across matching route_type
    explicitly excludes the route itself from its own peer calculation,
    and ignores routes of different route_type.
    """
    # In week '2024-02-05':
    # Route 1: R1 (Short) -> cost = 2.0
    # Route 2: R2 (Short) -> cost = 4.0
    # Route 3: R3 (Short) -> cost = 6.0
    # Route 4: R4 (Long)  -> cost = 100.0 (different route_type!)
    rows = [
        {"route": "R1", "origin": "A", "destination": "B", "route_type": "Short", "week_of": "2024-02-05",
         "total_cost": 2000.0, "total_tonne_km": 1000.0, "cost_per_tonne_km": 2.0, "shipment_count": 2},
        {"route": "R2", "origin": "C", "destination": "D", "route_type": "Short", "week_of": "2024-02-05",
         "total_cost": 4000.0, "total_tonne_km": 1000.0, "cost_per_tonne_km": 4.0, "shipment_count": 2},
        {"route": "R3", "origin": "E", "destination": "F", "route_type": "Short", "week_of": "2024-02-05",
         "total_cost": 6000.0, "total_tonne_km": 1000.0, "cost_per_tonne_km": 6.0, "shipment_count": 2},
        {"route": "R4", "origin": "G", "destination": "H", "route_type": "Long", "week_of": "2024-02-05",
         "total_cost": 100000.0, "total_tonne_km": 1000.0, "cost_per_tonne_km": 100.0, "shipment_count": 2},
    ]
    weekly = pd.DataFrame(rows)

    all_metrics, _ = compute_baselines_and_flags(weekly, spike_threshold_pct=20.0)

    # For R1 (Short):
    # Peers are R2 (4.0) and R3 (6.0). R1 (2.0) is excluded! R4 (Long) is excluded!
    # Expected peer average = (4.0 + 6.0) / 2 = 5.0
    r1 = all_metrics[all_metrics['route'] == 'R1'].iloc[0]
    assert np.isclose(r1['peer_avg'], 5.0), f"Expected 5.0, got {r1['peer_avg']}"
    assert r1['peers_used'] == 2

    # For R2 (Short):
    # Peers are R1 (2.0) and R3 (6.0). Expected peer average = (2.0 + 6.0) / 2 = 4.0
    r2 = all_metrics[all_metrics['route'] == 'R2'].iloc[0]
    assert np.isclose(r2['peer_avg'], 4.0), f"Expected 4.0, got {r2['peer_avg']}"
    assert r2['peers_used'] == 2

    # For R4 (Long):
    # There are NO other Long routes in this week.
    # Peer average must be NaN
    r4 = all_metrics[all_metrics['route'] == 'R4'].iloc[0]
    assert np.isnan(r4['peer_avg']), "Expected NaN peer_avg when no other peers exist in same route_type"
    assert r4['peers_used'] == 0
    assert r4['vs_similar_routes'] == "No peer routes available this week"
