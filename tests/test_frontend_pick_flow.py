from pathlib import Path


INDEX_HTML = Path(__file__).parents[1] / "public" / "index.html"


def _pick_bottle_source() -> str:
    page = INDEX_HTML.read_text(encoding="utf-8")
    start = page.index("async function pickBottle()")
    end = page.index("async function openHistory()", start)
    return page[start:end]


def test_pick_bottle_uses_legacy_endpoint_and_flat_response():
    source = _pick_bottle_source()

    assert "authFetch(`/api/pick?${params.toString()}`)" in source
    assert "/api/pick/full" not in source
    assert "if (data.hasBottle)" in source
    assert "pickResult.value = data;" in source
    assert "data.matchReason" in source
    assert "data.matchLevel" in source
    assert "data.hasMatch" not in source
    assert "const matchData" not in source


def test_pick_result_does_not_render_full_workflow_only_fields():
    page = INDEX_HTML.read_text(encoding="utf-8")

    assert "pickResult.compatibility" not in page
    assert "pickResult.iceBreakHints" not in page
