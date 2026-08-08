"""Shared pytest fixtures.

The Tk root lives here, at **session** scope, because it may only be created
once per process. Each module used to own a module-scoped ``tk.Tk()``; the
second one to run raised ``invalid command name "tcl_findLibrary"`` -- Tcl's
init state does not survive the first interpreter being destroyed -- and that
TclError is indistinguishable from having no display, so the later module
silently *skipped* its tests instead of reporting. Six real test_plot_scroll
tests were skipping in every full-suite run for exactly that reason.

Tests take a ``Toplevel`` off this root; they must not destroy the root itself.
"""

import pytest

tk = pytest.importorskip('tkinter')


@pytest.fixture(scope='session')
def tk_root():
    """One Tk interpreter for the whole session."""
    try:
        r = tk.Tk()
    except tk.TclError as exc:                  # genuinely no display
        pytest.skip(f'no Tk display available: {exc}')
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass
