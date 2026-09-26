"""Per-screen UI profile for the local (no-API) screen readers.

Colonist.io's layout depends on the window size, zoom and client version, and the synthetic
renderer only imitates it.  A :class:`UiProfile` records what was measured on the user's own
screen so the readers stop guessing:

* ``regions`` - boxes of the UI panels (``log``, ``player_panel``, ``hand_bar``, ``dice``,
  ``bank``) as fractions ``(x0, y0, x1, y1)`` of the screen, so one profile works for any
  capture of the same window layout at another resolution;
* ``ocr`` - what the game-log reader (:mod:`catanbot.vision.logocr`) learned on this screen
  (colours, font, exemplars).  The log reader owns the contents of this dict; exemplar arrays
  live in a sidecar ``.npz`` next to the JSON file (``ocr["exemplars"]`` holds its file name).

The JSON file is written atomically and refuses to overwrite a file that is not a profile.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

__all__ = ["REGION_NAMES", "PROFILE_FORMAT", "UiProfile", "ProfileError", "parse_box"]

#: Region names a profile may define.
REGION_NAMES: Tuple[str, ...] = ("log", "player_panel", "hand_bar", "dice", "bank")
PROFILE_FORMAT = "catanbot ui profile"
Box = Tuple[float, float, float, float]


class ProfileError(ValueError):
    """A profile file that cannot be read or would be overwritten wrongly."""


def _check_fraction_box(name: str, box: Sequence[float]) -> Box:
    if len(box) != 4:
        raise ProfileError(f"region '{name}': expected 4 numbers x0,y0,x1,y1, got {list(box)}")
    x0, y0, x1, y1 = (float(v) for v in box)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ProfileError(f"region '{name}': fractions must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, "
                           f"got {x0:g},{y0:g},{x1:g},{y1:g}")
    return (x0, y0, x1, y1)


def parse_box(text: str, size: Optional[Tuple[int, int]] = None) -> Box:
    """``"x,y,w,h"`` in pixels (``size`` needed to convert) or ``"x0,y0,x1,y1"`` fractions (all <= 1)
    -> a fraction box ``(x0, y0, x1, y1)``."""
    try:
        vals = [float(v) for v in str(text).replace(" ", "").split(",")]
    except ValueError:
        raise ProfileError(f"bad box '{text}': expected four numbers separated by commas")
    if len(vals) != 4:
        raise ProfileError(f"bad box '{text}': expected four numbers separated by commas")
    if all(0.0 <= v <= 1.0 for v in vals):
        return _check_fraction_box("box", vals)
    if size is None:
        raise ProfileError(f"box '{text}' is in pixels (x,y,w,h): the screen size is needed to use it")
    w, h = size
    x, y, bw, bh = vals
    if bw <= 0 or bh <= 0:
        raise ProfileError(f"bad box '{text}': width and height must be positive (x,y,w,h in pixels)")
    x0, y0 = max(0.0, x / w), max(0.0, y / h)
    x1, y1 = min(1.0, (x + bw) / w), min(1.0, (y + bh) / h)
    return _check_fraction_box("box", (x0, y0, x1, y1))


@dataclass
class UiProfile:
    """Regions (screen fractions) and log-reader data measured on one screen layout."""

    regions: Dict[str, Box] = field(default_factory=dict)
    screen: Optional[Tuple[int, int]] = None      # (width, height) the profile was measured on
    ocr: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None                    # file it was loaded from / saved to

    def set_region(self, name: str, box: Sequence[float]) -> None:
        if name not in REGION_NAMES:
            raise ProfileError(f"unknown region '{name}' (known: {', '.join(REGION_NAMES)})")
        self.regions[name] = _check_fraction_box(name, box)

    def region(self, name: str) -> Optional[Box]:
        return self.regions.get(name)

    def pixel_box(self, name: str, size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Region ``name`` in pixels ``(x0, y0, x1, y1)`` (end exclusive) for an image of ``size``."""
        box = self.regions.get(name)
        if box is None:
            return None
        w, h = size
        x0, y0, x1, y1 = _check_fraction_box(name, box)    # regions assigned directly bypass set_region
        px0 = min(w - 1, max(0, int(round(x0 * w))))
        py0 = min(h - 1, max(0, int(round(y0 * h))))
        return (px0, py0, min(w, max(px0 + 1, int(round(x1 * w)))), min(h, max(py0 + 1, int(round(y1 * h)))))

    def sidecar_path(self, name: str) -> Optional[str]:
        """Absolute path of a sidecar file ``name`` stored next to the profile JSON."""
        if not self.path:
            return None
        return os.path.join(os.path.dirname(os.path.abspath(self.path)), name)

    def to_dict(self) -> Dict[str, Any]:
        return {"format": PROFILE_FORMAT, "version": 1,
                "screen": list(self.screen) if self.screen else None,
                "regions": {k: list(v) for k, v in sorted(self.regions.items())},
                "ocr": self.ocr}

    @classmethod
    def from_dict(cls, d: Dict[str, Any], path: Optional[str] = None) -> "UiProfile":
        if not isinstance(d, dict) or d.get("format") != PROFILE_FORMAT:
            raise ProfileError(f"{path or 'data'} is not a catanbot UI profile")
        prof = cls(path=path)
        for name, box in (d.get("regions") or {}).items():
            prof.set_region(name, box)
        scr = d.get("screen")
        prof.screen = (int(scr[0]), int(scr[1])) if scr else None
        prof.ocr = dict(d.get("ocr") or {})
        return prof

    @classmethod
    def load(cls, path: str) -> "UiProfile":
        try:
            with open(path) as f:
                d = json.load(f)
        except json.JSONDecodeError as ex:
            raise ProfileError(f"{path}: not valid JSON ({ex})")
        return cls.from_dict(d, path=path)

    @classmethod
    def open(cls, path: Optional[str]) -> "UiProfile":
        """The profile at ``path``, or an empty one (to be saved there) when the file does not exist."""
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            return cls.load(path)
        return cls(path=path)

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path
        if not path:
            raise ProfileError("no path to save the profile to")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path) as f:
                    old = json.load(f)
            except (OSError, json.JSONDecodeError):
                old = None
            if not isinstance(old, dict) or old.get("format") != PROFILE_FORMAT:
                raise ProfileError(f"{path} exists and is not a catanbot UI profile: refusing to overwrite it")
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, path)
        self.path = path
        return path
