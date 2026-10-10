from __future__ import annotations

from hoops.bench import bench, format_table


def test_bench_reports_rows_and_errors():
    rows = bench(("fake:scripted", "nope:missing"), sizes=(320,), n_frames=4, warmup=1)
    ok, bad = rows
    assert ok["detector"] == "fake:scripted" and ok["fps"] > 0 and ok["ms_per_frame"] >= 0
    assert ok["device"] == "cpu" and ok["n_frames"] == 4
    assert "unknown detector" in bad["error"]
    table = format_table(rows)
    assert "fake:scripted" in table and "error" in table
