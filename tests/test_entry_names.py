from types import SimpleNamespace

import pytest

from iscdc.entry_names import fallback_entry_name, load_entry_names, resolve_entry_names


def _dataset(entry_id="A", **overrides):
    values = dict(
        entry_id=entry_id,
        dataset_type="full",
        title="One slide",
        organism="Mouse",
        tissue="Brain",
        modalities=[SimpleNamespace(technology="Xenium")],
    )
    return SimpleNamespace(**(values | overrides))


def test_shipped_names_cover_approved_entries_with_distinct_readable_names():
    names = load_entry_names()
    assert len(names) == 40
    assert len({name.casefold() for name in names.values()}) == 40
    assert names["S044"] == "Xenium Human Head and Neck Tumors — RNA and TCR"
    assert all(name != entry_id for entry_id, name in names.items())


@pytest.mark.parametrize("content", ["[broken", "- list", "A: first\nA: second", "? [a, b]\n: x"])
def test_malformed_name_file_falls_back_with_warning(tmp_path, caplog, content):
    path = tmp_path / "names.yaml"
    path.write_text(content)
    assert load_entry_names(path) == {}
    assert "using fallback names" in caplog.text


def test_missing_name_file_falls_back_with_warning(tmp_path, caplog):
    assert load_entry_names(tmp_path / "missing.yaml") == {}
    assert "using fallback names" in caplog.text


def test_invalid_rows_do_not_hide_valid_names(tmp_path, caplog):
    path = tmp_path / "names.yaml"
    path.write_text('A: Good name\nB: " "\nC: 123\nD: " padded "\n../E: Unsafe\n')
    assert load_entry_names(path) == {"A": "Good name"}
    assert caplog.text.count("Ignoring invalid Entry name") == 4


def test_fallback_uses_complete_group_and_is_independent_of_member_order():
    first = _dataset()
    second = _dataset(
        organism=["Human", "Mouse"],
        tissue=["Lung", "Brain"],
        modalities=[SimpleNamespace(technology=["SPOTS", "Xenium"])],
    )
    assert fallback_entry_name([first]) == "One slide"
    expected = "SPOTS, Xenium — Human, Mouse — Brain, Lung"
    assert fallback_entry_name([first, second]) == expected
    assert fallback_entry_name([second, first]) == expected


def test_resolved_names_only_include_full_entries_and_warn_about_missing_names(tmp_path, caplog):
    path = tmp_path / "names.yaml"
    path.write_text("A: Curated name\nUnused: Not in catalogue\n")
    names = resolve_entry_names(
        [_dataset(), _dataset("B"), _dataset("C", dataset_type="train")], path
    )
    assert names == {"A": "Curated name", "B": "One slide"}
    assert "No configured name for Entry B" in caplog.text
