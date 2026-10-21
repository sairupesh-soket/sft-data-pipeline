"""Build docs/SFT_PIPELINE.pdf.   python3 docs/build_pdf.py
Text: docs/pipeline_content.py (ASD-STE100). Examples: tests/cases.py run through the real code.
Validation rules: docs/validation_rules.py."""
import copy
import json
import os
import sys
import textwrap
from collections import Counter
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                Preformatted, PageBreak, KeepTogether, CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(D)
sys.path.insert(0, REPO); sys.path.insert(0, D); sys.path.insert(0, os.path.join(REPO, "tests"))
os.environ.setdefault("SFTPIPE_SRC", "."); os.environ.setdefault("SFTPIPE_OUT", "/tmp/sftpipe_doc_out")
import pipeline_content as C          # noqa: E402
import validation_rules as V          # noqa: E402
from cases import REPAIR_CASES, ADAPTER_CASES   # noqa: E402
from test_pipeline import run_repair, run_adapter   # noqa: E402

FONT_DIR = os.environ.get("DOC_FONT_DIR", "/usr/share/fonts/truetype/dejavu/")
pdfmetrics.registerFont(TTFont("DV", FONT_DIR + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", FONT_DIR + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVI", FONT_DIR + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DVM", FONT_DIR + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

INK, MUTED, LINE, BG = colors.HexColor("#1b1b1a"), colors.HexColor("#5a5955"), colors.HexColor("#d9d8d3"), colors.HexColor("#f2f1ee")
BLUE = colors.HexColor("#2a78d6")
IN_BG, IN_LN = colors.HexColor("#fdecea"), colors.HexColor("#c62828")
OUT_BG, OUT_LN = colors.HexColor("#e8f5e9"), colors.HexColor("#2e7d32")
ACT = {"FLAG": colors.HexColor("#c62828"), "INFO": colors.HexColor("#5a5955")}

ss = {
    "title": ParagraphStyle("title", fontName="DVB", fontSize=26, leading=32, textColor=INK),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=16, leading=21, textColor=INK, spaceBefore=4, spaceAfter=6),
    "toch": ParagraphStyle("toch", fontName="DVB", fontSize=16, leading=21, textColor=INK, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=11.5, leading=15, textColor=INK, spaceBefore=6, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.3, leading=13.2, textColor=INK, spaceAfter=4),
    "bul": ParagraphStyle("bul", fontName="DV", fontSize=9.1, leading=12.8, textColor=INK, leftIndent=10, bulletIndent=0, spaceAfter=2),
    "lab": ParagraphStyle("lab", fontName="DVB", fontSize=8.2, leading=11, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.6, leading=11.6, textColor=INK),
    "cellm": ParagraphStyle("cellm", fontName="DVM", fontSize=7.8, leading=10.4, textColor=INK),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=7.4, leading=9.4, textColor=INK),
    "note": ParagraphStyle("note", fontName="DVI", fontSize=8.8, leading=12, textColor=MUTED, spaceAfter=4),
    "toc1": ParagraphStyle("toc1", fontName="DV", fontSize=10, leading=15),
}
W = A4[0] - 36 * mm


def P(t, s="body"):
    return Paragraph(escape(str(t)).replace("\n", "<br/>"), ss[s])


def wrap(t, width):
    out = []
    for line in str(t).split("\n"):
        if len(line) <= width:
            out.append(line); continue
        ind = len(line) - len(line.lstrip())
        out += textwrap.wrap(line, width, subsequent_indent=" " * (ind + 2), break_long_words=True,
                             replace_whitespace=False, drop_whitespace=False) or [""]
    return "\n".join(out)


def code_box(t, width=100):
    tb = Table([[Preformatted(wrap(t, width), ss["code"])]], colWidths=[W])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BG), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return tb


def table(header, rows, widths=None):
    data = [[P(h, "lab") for h in header]] + [[P(c, "cell") for c in r] for r in rows]
    tb = Table(data, colWidths=widths or [W / len(header)] * len(header), repeatRows=1)
    tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                            ("BACKGROUND", (0, 0), (-1, 0), BG), ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4)]))
    return tb


def blocks(bl):
    out = []
    for b in bl:
        k = b[0]
        if k == "p": out.append(P(b[1]))
        elif k == "h2": out.append(Paragraph(escape(b[1]), ss["h2"]))
        elif k == "note": out.append(P(b[1], "note"))
        elif k == "code": out += [code_box(b[1]), Spacer(1, 5)]
        elif k == "b":
            out += [Paragraph(escape(x), ss["bul"], bulletText="•") for x in b[1]]
            out.append(Spacer(1, 4))
        elif k == "table":
            n = len(b[1])
            widths = [W * 0.22] + [W * 0.78 / (n - 1)] * (n - 1) if n > 1 else None
            out += [table(b[1], b[2], widths), Spacer(1, 6)]
    return out


def jrow(o):
    """OpenAI row as JSON, one message per line (still valid JSON)."""
    if isinstance(o, str):
        return o
    if isinstance(o, dict) and isinstance(o.get("messages"), list) and all(isinstance(m, dict) for m in o["messages"]):
        parts = ['{"messages": [']
        ms = o["messages"]
        for i, m in enumerate(ms):
            parts.append("  " + json.dumps(m, ensure_ascii=False) + ("," if i < len(ms) - 1 else ""))
        rest = [k for k in o if k != "messages"]
        parts.append("]" + ("," if rest else "}"))
        for j, k in enumerate(rest):
            v = o[k]
            if k == "tools" and isinstance(v, list):
                parts.append(' "tools": [')
                for i, tl in enumerate(v):
                    parts.append("  " + json.dumps(tl, ensure_ascii=False) + ("," if i < len(v) - 1 else ""))
                parts.append(" ]" + ("," if j < len(rest) - 1 else "}"))
            else:
                parts.append(f' "{k}": ' + json.dumps(v, ensure_ascii=False) + ("," if j < len(rest) - 1 else "}"))
        return "\n".join(parts)
    if isinstance(o, list):
        return "[\n" + ",\n".join("  " + json.dumps(m, ensure_ascii=False) for m in o) + "\n]"
    return json.dumps(o, ensure_ascii=False, indent=1)


def pair(label_a, a, label_b, b, width=51):
    ta = jrow(a)
    tb_ = jrow(b)
    t = Table([[P(label_a, "lab"), P(label_b, "lab")],
               [Preformatted(wrap(ta, width), ss["code"]), Preformatted(wrap(tb_, width), ss["code"])]], colWidths=[W / 2, W / 2])
    t.setStyle(TableStyle([("BACKGROUND", (0, 1), (0, 1), IN_BG), ("BACKGROUND", (1, 1), (1, 1), OUT_BG),
                           ("LINEBEFORE", (0, 1), (0, 1), 2.2, IN_LN), ("LINEBEFORE", (1, 1), (1, 1), 2.2, OUT_LN),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 1), (-1, 1), 5),
                           ("BOTTOMPADDING", (0, 1), (-1, 1), 5), ("LEFTPADDING", (0, 1), (-1, 1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def diagram():
    steps = [("Raw sources", "parquet / jsonl in many schemas"), ("1. Convert", "adapters -> OpenAI messages"),
             ("2. Clean", "repair steps (new output folder)"), ("3. Validate", "67 rules: flag rows, change nothing"),
             ("4. Stats", "rows, tokens, turns, tools, reasoning")]
    bw, bh, gap = 150, 30, 14
    h = len(steps) * (bh + gap)
    d = Drawing(W, h)
    x = (W - bw) / 2
    for i, (t, sub) in enumerate(steps):
        y = h - (i + 1) * (bh + gap) + gap
        fill = BG if i == 0 else colors.HexColor("#e3eefb")
        d.add(Rect(x, y, bw, bh, rx=5, ry=5, fillColor=fill, strokeColor=BLUE if i else MUTED, strokeWidth=1))
        d.add(String(x + bw / 2, y + bh - 12, t, fontName="DVB", fontSize=9, textAnchor="middle", fillColor=INK))
        d.add(String(x + bw / 2, y + 6, sub, fontName="DV", fontSize=7, textAnchor="middle", fillColor=MUTED))
        if i < len(steps) - 1:
            ax = x + bw / 2
            d.add(Line(ax, y, ax, y - gap + 3, strokeColor=MUTED, strokeWidth=1))
            d.add(Polygon([ax - 3, y - gap + 5, ax + 3, y - gap + 5, ax, y - gap + 1], fillColor=MUTED, strokeColor=MUTED))
    d.add(String(x + bw + 10, h - 2.5 * (bh + gap) + gap + 12, "logs -> <out>_logs/", fontName="DV", fontSize=7, fillColor=MUTED))
    return d


def rule_block(r, num):
    a = r["label"]
    head = Table([[Paragraph(f'<b>{num}&nbsp;&nbsp;{escape(r["title"])}</b>', ss["h2"]),
                   Paragraph(f'<font color="white"><b>{a}</b></font>', ParagraphStyle("bd", parent=ss["cell"], alignment=TA_CENTER))]],
                 colWidths=[W - 26 * mm, 26 * mm])
    head.setStyle(TableStyle([("BACKGROUND", (1, 0), (1, 0), ACT[a]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    rid = r["id"] + ("_<field>" if r["id"] == "EKA_T1" else "_<name>" if r["id"] == "IDENTITY" else "_<details>" if r["id"] in ("RENDER", "FORMAT_bad_shape") else "")
    rows = [[P("Rule id", "lab"), Paragraph(f'<font name="DVM">{escape(rid)}</font>', ss["cell"])],
            [P("What it finds", "lab"), P(r["what"], "cell")], [P("Why it is a problem", "lab"), P(r["why"], "cell")],
            [P("How it is checked", "lab"), P(r["how"], "cell")]]
    if r.get("path"):
        rows.append([P("Example file", "lab"), Paragraph(f'<font name="DVM">{escape(r["path"])}</font>', ss["cell"])])
    info = Table(rows, colWidths=[30 * mm, W - 30 * mm])
    info.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
                              ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
    left = r["show"] or r["flagged"]
    if a == "INFO":
        ex = pair("EXAMPLE (correct, information only)", left, "CORRECT FORM", r["correct"])
    else:
        ex = pair("FLAGGED", left, "CORRECT FORM", r["correct"])
    return KeepTogether([head, Spacer(1, 2), info, Spacer(1, 3), ex, Spacer(1, 10)])


def rules_chapter(prefix):
    out = []
    groups = []
    for g, r in V.R:
        if not groups or groups[-1][0] != g:
            groups.append((g, []))
        groups[-1][1].append(r)
    for gi, (g, rules) in enumerate(groups, 1):
        out += [CondPageBreak(60 * mm), Paragraph(escape(f"{prefix}.{gi}  {g}"), ss["h2"])]
        for ri, r in enumerate(rules, 1):
            out.append(rule_block(r, f"{prefix}.{gi}.{ri}"))
    return out


class Doc(BaseDocTemplate):
    def __init__(self, fn):
        super().__init__(fn, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=16 * mm,
                         title="SFT Data Pipeline", author="SFT data team", subject="Convert, clean, validate SFT data (ASD-STE100)")
        self.addPageTemplates([PageTemplate(id="p", frames=[Frame(self.leftMargin, self.bottomMargin, self.width, self.height)],
                                            onPage=self.deco)])

    def deco(self, c, d):
        if d.page == 1:
            return
        c.saveState(); c.setFont("DV", 7.5); c.setFillColor(MUTED)
        c.drawString(18 * mm, A4[1] - 11 * mm, self.title)
        c.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {d.page}")
        c.setStrokeColor(LINE); c.line(18 * mm, A4[1] - 12.5 * mm, A4[0] - 18 * mm, A4[1] - 12.5 * mm)
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name == "h1":
            self.notify("TOCEntry", (0, f.getPlainText(), self.page))


def H1(t):
    return [CondPageBreak(50 * mm), Paragraph(escape(t), ss["h1"])]


def build(out):
    st = [Spacer(1, 60 * mm), Paragraph("SFT Data Pipeline", ss["title"]), PageBreak(),
          Paragraph("Contents", ss["toch"])]
    toc = TableOfContents(); toc.levelStyles = [ss["toc1"]]
    st += [toc, PageBreak()]

    st += H1("1. Overview") + [diagram(), Spacer(1, 8)] + blocks(C.OVERVIEW)
    st += [PageBreak()] + H1("2. Data format") + blocks(C.FORMAT)

    # 3 adapters
    st += [PageBreak()] + H1("3. Convert: source adapters") + blocks(C.ADAPTERS_INTRO)
    st += [Paragraph("Normalization examples (output of the code)", ss["h2"])]
    for title, fn, inp, exp in ADAPTER_CASES:
        got = run_adapter(fn, inp)
        assert got == exp, title
        st += [KeepTogether([P(f"{title}   ({fn})", "lab"), Spacer(1, 2), pair("INPUT", inp, "OUTPUT", got), Spacer(1, 8)])]
    for a in C.ADAPTERS:
        st += [CondPageBreak(40 * mm), Paragraph(escape("Adapter: " + a["name"]), ss["h2"]),
               table(["Item", "Text"], [["Module", a["module"]], ["Input", a["input"]]], [W * 0.18, W * 0.82]), Spacer(1, 4)]
        st += [Paragraph(escape(x), ss["bul"], bulletText="•") for x in a["what"]]
        st += [Spacer(1, 3), code_box(a["cmd"]), Spacer(1, 8)]
    import importlib, ast
    A = importlib.import_module("sft_pipeline.adapters.agentic")
    asrc = open(os.path.join(REPO, "sft_pipeline", "adapters", "agentic.py")).read()
    expr = {}
    for node in ast.walk(ast.parse(asrc)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CONVERT" for t in node.targets) and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                expr[ast.literal_eval(k)] = ast.get_source_segment(asrc, v)
    fdoc = {n.name: (ast.get_docstring(n) or "").strip().split("\n")[0] for n in ast.walk(ast.parse(asrc)) if isinstance(n, ast.FunctionDef)}
    rows = []
    for ds in A.CONVERT:
        e = expr.get(ds, "")
        fname = e.split("(")[0] if e else ""
        rows.append([ds, e, fdoc.get(fname, "")])
    st += [CondPageBreak(40 * mm), Paragraph("Agentic adapter: dataset table", ss["h2"]),
           P("Each dataset has its own converter function. The first line of the function docstring tells the mapping."),
           table(["Dataset", "Converter call", "Converter description"], rows, [W * 0.36, W * 0.30, W * 0.34]), Spacer(1, 8),
           Paragraph("Agentic adapter: datasets that are not converted", ss["h2"]),
           table(["Dataset", "Reason"], [[k, v] for k, v in A.SKIP.items()], [W * 0.5, W * 0.5]), Spacer(1, 8)]
    import ast
    rows = []
    for i in range(1, 10):
        src = open(os.path.join(REPO, "sft_pipeline", "adapters", "traces", f"extra{i}.py")).read()
        d0 = (ast.get_docstring(ast.parse(src)) or "").split("\n\n")[0].replace("\n", " ")
        rows.append([f"traces_extra{i}", d0])
    st += [CondPageBreak(40 * mm), Paragraph("Traces adapter: extra modules", ss["h2"]),
           table(["Adapter", "Datasets (from the module docstring)"], rows, [W * 0.2, W * 0.8])]

    # 4 repairs
    st += [PageBreak()] + H1("4. Clean: repair steps") + blocks(C.REPAIRS_INTRO)
    by_step = {}
    for step, title, inp, exp in REPAIR_CASES:
        by_step.setdefault(step, []).append((title, inp, exp))
    order = ["format", "shapes", "hermes", "tool_ids", "think_content", "move_think", "move_other_tags",
             "harness_fallback", "final", "role_rules", "text_drops"]
    for i, step in enumerate(order, 1):
        name, mod, lines = C.REPAIRS[step]
        st += [CondPageBreak(60 * mm), Paragraph(escape(f"4.{i}  {step}  -  {name}"), ss["h2"]), P(f"Module: sft_pipeline/{mod}", "note")]
        st += [Paragraph(escape(x), ss["bul"], bulletText="•") for x in lines]
        st.append(Spacer(1, 3))
        for title, inp, exp in by_step.get(step, []):
            got = run_repair(step, inp)
            assert got == exp, title
            right = got if not isinstance(got, str) else f"(row not written)\nreason: {got[5:]}"
            st += [KeepTogether([P("Example: " + title, "lab"), Spacer(1, 2), pair("INPUT", inp, "OUTPUT", right), Spacer(1, 8)])]

    st += [PageBreak()] + H1("5. Production recipe for each source family") + blocks(C.RECIPES_INTRO)
    for name, cmds, order in C.RECIPES:
        st += [CondPageBreak(45 * mm), Paragraph(escape(name), ss["h2"]), code_box("\n".join(cmds)),
               Spacer(1, 3), P("Clean step order: " + order, "note"), Spacer(1, 6)]
    st += [Paragraph("End-to-end test against the production data", ss["h2"]),
           table(C.E2E_TABLE[0], C.E2E_TABLE[1], [W * 0.14, W * 0.36, W * 0.2, W * 0.3]), Spacer(1, 8)]
    st += H1("6. hf_domain tools") + blocks(C.HF_DOMAIN)
    st += H1("7. Validate: rules (flag only)") + blocks(C.VALIDATE)
    st += [P("Labels: FLAG = the row has the problem and is in the issue list. INFO = information only. "
             "Every example below is checked by tests/test_rule_examples.py.", "note")]
    st += rules_chapter("7")
    st += [PageBreak()] + H1("8. Stats") + blocks(C.STATS)
    st += H1("9. Legacy code") + blocks(C.LEGACY)
    st += H1("10. Steps of other teams") + blocks(C.EXTERNAL)
    st += H1("11. How to add a new source") + blocks(C.NEW_SOURCE)
    st += H1("12. Lessons") + [P(f"{i}. {l}") for i, l in enumerate(C.LESSONS, 1)]
    st += H1("13. What the pipeline does not check") + blocks(C.NOT_CHECKED)
    Doc(out).multiBuild(st)


if __name__ == "__main__":
    out = os.path.join(D, "SFT_PIPELINE.pdf")
    build(out); print(out)
