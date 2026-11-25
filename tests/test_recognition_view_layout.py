from pathlib import Path
from types import SimpleNamespace
import sys
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desktop.views.recognition_view import RecognitionView


class RecognitionViewLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
            cls.root.withdraw()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk is unavailable: {error}") from error

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.view = RecognitionView(self.root)
        self.view.pack(fill="both", expand=True)
        self.root.update_idletasks()

    def tearDown(self):
        self.view.destroy()
        self.root.update_idletasks()

    def test_details_panel_has_vertical_scrollbar_and_top_controls(self):
        scroller = self.view.details_scroller

        self.assertEqual(str(scroller.scrollbar.cget("orient")), "vertical")
        self.assertEqual(scroller.scrollbar.winfo_manager(), "grid")
        self.assertEqual(int(scroller.scrollbar.grid_info()["column"]), 1)
        self.assertIs(self.view.start_button.master, scroller.content)
        self.assertIs(self.view.stop_button.master, scroller.content)
        self.assertIs(self.view.record_trial_button.master, scroller.content)

    def test_all_diagnostic_labels_are_in_scrollable_content(self):
        variables = {
            str(child.cget("textvariable"))
            for child in self.view.details_scroller.content.winfo_children()
            if isinstance(child, ttk.Label)
            and child.cget("textvariable")
        }

        self.assertIn(str(self.view.quality_diagnostics), variables)
        self.assertIn(str(self.view.matcher_diagnostics), variables)
        self.assertIn(str(self.view.recognition_status), variables)
        self.assertIn(str(self.view.processing_time), variables)

    def test_teacher_review_controls_are_in_right_scrollable_panel(self):
        scroller = self.view.review_scroller

        self.assertEqual(str(scroller.scrollbar.cget("orient")), "vertical")
        self.assertEqual(scroller.scrollbar.winfo_manager(), "grid")
        self.assertEqual(int(scroller.scrollbar.grid_info()["column"]), 1)
        self.assertIs(self.view.review_decision_combo.master, scroller.content)
        self.assertIs(self.view.review_note_entry.master, scroller.content)
        self.assertIs(self.view.save_review_button.master, scroller.content)
        self.assertIs(self.view.finalize_button.master, scroller.content)
        self.assertIs(self.view.recent_events.master, scroller.content)

    def test_teacher_review_scroller_tracks_width_and_content_height(self):
        scroller = self.view.review_scroller

        scroller._on_canvas_configure(SimpleNamespace(width=280))
        self.root.update_idletasks()
        scroller._on_content_configure()

        self.assertEqual(
            float(scroller.canvas.itemcget(scroller._content_window, "width")),
            280,
        )
        self.assertTrue(scroller.canvas.cget("scrollregion"))

    def test_teacher_review_mousewheel_scrolls_only_its_canvas(self):
        scroller = self.view.review_scroller
        scroll = Mock()
        scroller.canvas.yview_scroll = scroll

        result = scroller._on_mousewheel(SimpleNamespace(delta=-120))

        scroll.assert_called_once_with(1, "units")
        self.assertEqual(result, "break")

    def test_resize_updates_content_width_and_scroll_region(self):
        scroller = self.view.details_scroller

        scroller._on_canvas_configure(SimpleNamespace(width=360))
        self.root.update_idletasks()
        scroller._on_content_configure()

        self.assertEqual(
            float(scroller.canvas.itemcget(
                scroller._content_window,
                "width",
            )),
            360,
        )
        self.assertTrue(scroller.canvas.cget("scrollregion"))

    def test_windows_mousewheel_scrolls_only_the_details_canvas(self):
        scroller = self.view.details_scroller
        scroll = Mock()
        scroller.canvas.yview_scroll = scroll

        result = scroller._on_mousewheel(SimpleNamespace(delta=-120))

        scroll.assert_called_once_with(1, "units")
        self.assertEqual(result, "break")


if __name__ == "__main__":
    unittest.main()
