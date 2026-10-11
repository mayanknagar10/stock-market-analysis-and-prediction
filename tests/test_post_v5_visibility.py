def test_final_window_and_boundary_crossing_outcomes_stay_hidden():
    from research.post_v5.reporting import visible_shadow_records
    records=[{'forecast_as_of':'2027-10-10T10:00:00Z','outcome':{'maturity_timestamp':'2027-10-20T10:00:00Z','actual_return':.3}},
        {'forecast_as_of':'2027-10-11T10:00:00Z','outcome':{'maturity_timestamp':'2027-10-12T10:00:00Z','actual_return':.1}}]
    visible,sealed=visible_shadow_records(records,'2027-10-11T00:00:00Z')
    assert sealed==1
    assert len(visible)==1 and visible[0]['outcome'] is None
    assert records[0]['outcome']['actual_return']==.3
