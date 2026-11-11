"""Human-readable evidence for a verdict: query vs enrolled spray with matched droplets."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .pipeline import Status, Verdict

_COLORS = {Status.GENUINE: (60, 200, 60), Status.MISMATCH: (0, 165, 255),
           Status.NOT_FOUND: (60, 60, 230), Status.UNREADABLE: (160, 160, 160)}


def render_verdict(verdict: Verdict, enrolled_fmap: np.ndarray | None, out_path: str | Path,
                   max_lines: int = 150) -> None:
    """Panels: query canonical crop | query spray map | enrolled spray map, inlier matches drawn."""
    color = _COLORS[verdict.status]
    if verdict.capture is None:
        canvas = np.full((130, 1100, 3), 30, np.uint8)
    else:
        q = verdict.capture.fingerprint.fmap
        n = q.shape[0]
        crop = cv2.cvtColor(verdict.capture.registration.crop, cv2.COLOR_RGB2BGR)
        enrolled = enrolled_fmap if enrolled_fmap is not None else np.zeros_like(q)
        panels = np.hstack([crop, cv2.cvtColor(q, cv2.COLOR_GRAY2BGR), cv2.cvtColor(enrolled, cv2.COLOR_GRAY2BGR)])
        ev = verdict.evidence
        if enrolled_fmap is not None and ev is not None and len(ev.match.pairs):
            pairs = ev.match.pairs
            pick = pairs[np.linspace(0, len(pairs) - 1, min(max_lines, len(pairs))).astype(int)]
            qpts = verdict.capture.fingerprint.local.points
            for ia, ib in pick:
                pa = tuple(int(v) for v in np.round(qpts[ia] + (n, 0)))
                pb = tuple(int(v) for v in np.round(ev.enrolled_points[ib] + (2 * n, 0)))
                cv2.line(panels, pa, pb, color, 1, cv2.LINE_AA)
                cv2.circle(panels, pa, 5, color, 1, cv2.LINE_AA)
                cv2.circle(panels, pb, 5, color, 1, cv2.LINE_AA)
        for i, label in enumerate(["query (canonical)", "query spray", "enrolled spray"]):
            cv2.putText(panels, label, (i * n + 12, n - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        canvas = np.vstack([np.full((110, panels.shape[1], 3), 30, np.uint8), panels])
    title = verdict.status.value + (f"  note {verdict.note_id}" if verdict.note_id else "")
    cv2.putText(canvas, title, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.putText(canvas, f"{verdict.inliers} matched droplets | cosine {verdict.cosine:.3f} | {verdict.message}",
                (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 1)
    cv2.imwrite(str(out_path), canvas)
