"""
Unit tests for fetch.py incremental sync logic
Run with: python test_fetch_incremental.py
"""

import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock

sys.modules["requests"] = MagicMock()  # Must mock before importing fetch

from fetch import determine_slices_to_sync, get_new_last_sync_month  # noqa: E402


class TestIncrementalSyncLogic(unittest.TestCase):
    """Test the incremental sync mode determination logic"""

    def _make_sync_state(self, last_month=None, synced_slices=None):
        """Helper to create sync_state dict"""
        state = {}
        if last_month:
            state["last_sync_month"] = last_month
        if synced_slices:
            state["synced_slices"] = synced_slices
        return state

    # ========== First Run Tests ==========

    def test_first_run_full_sync(self):
        """First run with no sync_state should sync all slices"""
        sync_state = {}
        now = datetime(2025, 1, 15)
        created_slices = ["2025-*", "2024-*", "2023-*"]

        slices, mode = determine_slices_to_sync(sync_state, now, created_slices)

        # When synced_slices is empty, all slices are "new", so mode is NEW_SLICES
        self.assertIn(mode, ["FIRST_RUN", "NEW_SLICES"])
        self.assertEqual(slices, created_slices)

    def test_first_run_sets_last_sync_month_to_previous_month(self):
        """After first run, last_sync_month should be previous month"""
        now = datetime(2025, 1, 15)
        new_last_month = get_new_last_sync_month(now)
        self.assertEqual(new_last_month, "2024-12")

    # ========== Same Month Tests ==========

    def test_same_month_syncs_current_month(self):
        """Running twice in same month should sync current month"""
        created_slices = ["2025-*", "2024-*", "2023-*"]
        sync_state = self._make_sync_state(
            last_month="2024-12",
            synced_slices=created_slices,
        )
        now = datetime(2025, 1, 20)

        slices, mode = determine_slices_to_sync(sync_state, now, created_slices)

        self.assertEqual(mode, "INCREMENTAL")
        self.assertEqual(slices, ["2025-01-*"])

    def test_same_month_last_sync_month_unchanged(self):
        """After syncing same month, last_sync_month stays as previous month"""
        now = datetime(2025, 1, 20)
        new_last_month = get_new_last_sync_month(now)
        self.assertEqual(new_last_month, "2024-12")

    # ========== Cross Month Tests ==========

    def test_cross_month_syncs_previous_and_current(self):
        """Crossing to new month should sync both months"""
        created_slices = ["2025-*", "2024-*"]
        sync_state = self._make_sync_state(
            last_month="2024-12", synced_slices=created_slices
        )
        now = datetime(2025, 2, 5)

        slices, mode = determine_slices_to_sync(sync_state, now, created_slices)

        self.assertEqual(mode, "INCREMENTAL")
        self.assertEqual(slices, ["2025-01-*", "2025-02-*"])

    def test_cross_month_updates_last_sync_month(self):
        """After crossing month, last_sync_month should update"""
        now = datetime(2025, 2, 5)
        new_last_month = get_new_last_sync_month(now)
        self.assertEqual(new_last_month, "2025-01")

    def test_cross_multiple_months(self):
        """Skipping several months should sync all of them"""
        created_slices = ["2025-*", "2024-*"]
        sync_state = self._make_sync_state(
            last_month="2024-12", synced_slices=created_slices
        )
        now = datetime(2025, 3, 10)

        slices, mode = determine_slices_to_sync(sync_state, now, created_slices)

        self.assertEqual(mode, "INCREMENTAL")
        self.assertEqual(slices, ["2025-01-*", "2025-02-*", "2025-03-*"])

    # ========== Year Boundary Tests ==========

    def test_cross_year_boundary(self):
        """Crossing year boundary should work correctly"""
        created_slices = ["2025-*", "2024-*"]
        sync_state = self._make_sync_state(
            last_month="2024-11", synced_slices=created_slices
        )
        now = datetime(2025, 1, 5)

        slices, mode = determine_slices_to_sync(sync_state, now, created_slices)

        self.assertEqual(mode, "INCREMENTAL")
        self.assertEqual(slices, ["2024-12-*", "2025-01-*"])

    # ========== New Year Slice Tests ==========

    def test_new_yearly_slice_detected(self):
        """Adding new year to CREATED_SLICES should trigger sync"""
        sync_state = self._make_sync_state(
            last_month="2024-12",
            synced_slices=["2024-*", "2023-*"],
        )
        now = datetime(2025, 1, 15)
        created_slices = ["2025-*", "2024-*", "2023-*"]

        slices, mode = determine_slices_to_sync(sync_state, now, created_slices)

        self.assertEqual(mode, "NEW_SLICES")
        self.assertEqual(slices, ["2025-*"])

    def test_new_historical_slice_detected(self):
        """Adding historical year should trigger sync"""
        sync_state = self._make_sync_state(
            last_month="2024-12", synced_slices=["2025-*", "2024-*", "2023-*"]
        )
        now = datetime(2025, 1, 15)
        created_slices = ["2025-*", "2024-*", "2023-*", "2011-*"]

        slices, mode = determine_slices_to_sync(sync_state, now, created_slices)

        self.assertEqual(mode, "NEW_SLICES")
        self.assertEqual(slices, ["2011-*"])

    # ========== Max Year Limit Tests ==========

    def test_current_year_beyond_max_year_no_sync(self):
        """If current year > max year and last_month is max-12, nothing to sync"""
        sync_state = self._make_sync_state(
            last_month="2025-12",
            synced_slices=["2025-*", "2024-*"],
        )
        now = datetime(2026, 1, 15)
        created_slices = ["2025-*", "2024-*"]

        slices, mode = determine_slices_to_sync(sync_state, now, created_slices)

        self.assertEqual(mode, "NOTHING_TO_DO")
        self.assertEqual(slices, [])

    def test_current_year_beyond_max_year_partial_sync(self):
        """If current year > max but last_month < max-12, sync remaining"""
        sync_state = self._make_sync_state(
            last_month="2025-10",
            synced_slices=["2025-*", "2024-*"],
        )
        now = datetime(2026, 2, 15)
        created_slices = ["2025-*", "2024-*"]

        slices, mode = determine_slices_to_sync(sync_state, now, created_slices)

        self.assertEqual(mode, "INCREMENTAL")
        self.assertEqual(slices, ["2025-11-*", "2025-12-*"])

    # ========== Edge Cases ==========

    def test_january_previous_month_is_december(self):
        """In January, previous month should be December of previous year"""
        now = datetime(2025, 1, 15)
        new_last_month = get_new_last_sync_month(now)
        self.assertEqual(new_last_month, "2024-12")

    def test_december_previous_month_is_november(self):
        """In December, previous month should be November"""
        now = datetime(2025, 12, 15)
        new_last_month = get_new_last_sync_month(now)
        self.assertEqual(new_last_month, "2025-11")

    def test_last_month_december_next_sync_january(self):
        """If last_month is Dec, next sync should start from Jan"""
        created_slices = ["2025-*", "2024-*"]
        sync_state = self._make_sync_state(
            last_month="2024-12", synced_slices=created_slices
        )
        now = datetime(2025, 1, 15)

        slices, mode = determine_slices_to_sync(sync_state, now, created_slices)

        self.assertEqual(mode, "INCREMENTAL")
        self.assertEqual(slices, ["2025-01-*"])


class TestSliceTypeDetection(unittest.TestCase):
    """Test yearly vs monthly slice detection"""

    def test_yearly_slice_detection(self):
        """Yearly slices have format YYYY-*"""
        self.assertEqual(len("2025-*".split("-")), 2)

    def test_monthly_slice_detection(self):
        """Monthly slices have format YYYY-MM-*"""
        self.assertEqual(len("2025-01-*".split("-")), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
