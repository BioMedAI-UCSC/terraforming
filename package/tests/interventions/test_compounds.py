"""Tests for src.interventions.compounds.

Covers:
  - COMPOUNDS registry completeness and positive-valued fields
  - get_compound happy path and KeyError on unknown name
  - list_compounds returns sorted names
"""

from __future__ import annotations

import pytest

from src.interventions.compounds import (
    COMPOUNDS,
    CompoundProperties,
    get_compound,
    list_compounds,
)


class TestCompoundProperties:

    def test_all_compounds_have_positive_molecular_weight(self):
        for name, props in COMPOUNDS.items():
            assert props.molecular_weight_g_mol > 0, (
                f"{name}: molecular_weight_g_mol must be > 0"
            )

    def test_all_compounds_have_positive_lifetime(self):
        for name, props in COMPOUNDS.items():
            assert props.atmospheric_lifetime_yr > 0, (
                f"{name}: atmospheric_lifetime_yr must be > 0"
            )

    def test_all_compounds_have_positive_rf_efficiency(self):
        for name, props in COMPOUNDS.items():
            assert props.rf_efficiency_W_m2_ppb > 0, (
                f"{name}: rf_efficiency_W_m2_ppb must be > 0"
            )

    def test_all_compounds_have_positive_gwp100(self):
        for name, props in COMPOUNDS.items():
            assert props.gwp100 > 0, f"{name}: gwp100 must be > 0"

    def test_all_compounds_have_non_empty_description(self):
        for name, props in COMPOUNDS.items():
            assert props.description, f"{name}: description must not be empty"

    def test_compound_properties_is_frozen(self):
        """CompoundProperties must be immutable (frozen dataclass)."""
        props = get_compound("CF4")
        with pytest.raises((AttributeError, TypeError)):
            props.gwp100 = 0  # type: ignore[misc]

    def test_known_compounds_present(self):
        """The seven key super-GHGs must be registered."""
        expected = {"CF4", "C2F6", "C3F8", "SF6", "NF3", "C4F10", "C6F14"}
        assert expected.issubset(set(COMPOUNDS))

    def test_cf4_rf_efficiency_marinova(self):
        """CF4 radiative forcing efficiency matches Marinova (2005) value."""
        props = get_compound("CF4")
        assert abs(props.rf_efficiency_W_m2_ppb - 0.0880) < 1e-4

    def test_sf6_rf_efficiency_marinova(self):
        """SF6 radiative forcing efficiency matches Marinova (2005) value."""
        props = get_compound("SF6")
        assert abs(props.rf_efficiency_W_m2_ppb - 0.5700) < 1e-4


class TestGetCompound:

    def test_returns_compound_properties_instance(self):
        result = get_compound("CF4")
        assert isinstance(result, CompoundProperties)

    def test_returns_correct_compound(self):
        props = get_compound("SF6")
        assert props.molecular_weight_g_mol == pytest.approx(146.1, rel=1e-3)

    def test_unknown_compound_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown compound"):
            get_compound("H2SO4_fake")

    def test_key_error_message_lists_available(self):
        """Error message should contain the available compound names."""
        with pytest.raises(KeyError) as exc_info:
            get_compound("NONEXISTENT")
        assert "CF4" in str(exc_info.value)

    def test_case_sensitive(self):
        """Lookup is case-sensitive; 'cf4' is not the same as 'CF4'."""
        with pytest.raises(KeyError):
            get_compound("cf4")


class TestListCompounds:

    def test_returns_list(self):
        result = list_compounds()
        assert isinstance(result, list)

    def test_list_is_sorted(self):
        result = list_compounds()
        assert result == sorted(result)

    def test_all_entries_in_registry(self):
        for name in list_compounds():
            assert name in COMPOUNDS

    def test_length_matches_registry(self):
        assert len(list_compounds()) == len(COMPOUNDS)
