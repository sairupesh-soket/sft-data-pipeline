"""Build docs/SFT_VALIDATION_RULES.pdf (validation only).   python3 docs/build_validation_pdf.py
Text and examples: docs/validation_rules.py (every example checked by tests/test_rule_examples.py)."""
import os
from xml.sax.saxutils import escape

import build_pdf as B
from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

V = B.V


def build(out):
    n = len(V.R); fl = sum(1 for g, r in V.R if r["label"] == "FLAG")
    groups = []
    for g, r in V.R:
        if g not in groups:
            groups.append(g)
    st = [Spacer(1, 50 * mm), Paragraph("SFT Validation Rules", B.ss["title"]), Spacer(1, 30 * mm)]
    sumt = Table([[B.P("Rules", "lab"), B.P("FLAG", "lab"), B.P("INFO", "lab"), B.P("Groups", "lab")],
                  [B.P(str(n), "h1"), B.P(str(fl), "h1"), B.P(str(n - fl), "h1"), B.P(str(len(groups)), "h1")]], colWidths=[B.W / 4] * 4)
    sumt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), B.BG), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6)]))
    st += [sumt, PageBreak(), Paragraph("Contents", B.ss["toch"])]
    toc = B.TableOfContents(); toc.levelStyles = [B.ss["toc1"]]
    st += [toc, PageBreak()] + B.H1("0. Introduction")
    for h, t in V.INTRO:
        st += [Paragraph(escape(h), B.ss["h2"]), B.P(t)]
    st += [PageBreak()] + B.H1("1. Rules") + B.rules_chapter("1")
    doc = B.Doc(out)
    doc.title = "SFT Validation Rules"
    doc.multiBuild(st)


if __name__ == "__main__":
    out = os.path.join(B.D, "SFT_VALIDATION_RULES.pdf")
    build(out); print(out)
