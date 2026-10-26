#!/usr/bin/env python3
"""Restyle the three original vector diagrams without flattening their artwork.

Read the original PDFs from --source-dir; never use a previously restyled PDF
as the source. Text is replaced at its original baseline, color, and size,
while all paths and image objects are retained. The two malformed character
mappings in insert-delete-example are corrected to the visually verified x
and m glyphs. No scientific values or diagram shapes are changed.
Requires PyMuPDF and the Linux Libertine O OpenType family used by the
official VLDB template. Fonts are found through --font-dir,
NORA_PAPER_FONT_DIR, or TeX Live's kpsewhich (in that order).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import pymupdf

NAMES = ("motivation-example", "NPA-repair-type", "insert-delete-example")
FACES = {"Regular": "LinLibertine_R.otf", "Bold": "LinLibertine_RB.otf",
         "Italic": "LinLibertine_RI.otf", "BoldItalic": "LinLibertine_RBI.otf"}


def find_font_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        font_dir = explicit
    elif os.environ.get("NORA_PAPER_FONT_DIR"):
        font_dir = Path(os.environ["NORA_PAPER_FONT_DIR"])
    else:
        try:
            regular = subprocess.check_output(
                ["kpsewhich", FACES["Regular"]], text=True).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            regular = ""
        if not regular:
            raise RuntimeError("Linux Libertine O fonts not found; set "
                               "NORA_PAPER_FONT_DIR or --font-dir.")
        font_dir = Path(regular).parent
    for filename in FACES.values():
        if not (font_dir / filename).is_file():
            raise FileNotFoundError(font_dir / filename)
    return font_dir


def restyle(source: Path, output: Path, font_dir: Path) -> dict:
    doc = pymupdf.open(source)
    before_paths = [len(p.get_drawings()) for p in doc]
    before_images = [len(p.get_images(full=True)) for p in doc]
    fonts = {face: pymupdf.Font(fontfile=str(font_dir / FACES[face]))
             for face in FACES}
    count = 0
    for page in doc:
        spans = [s for b in page.get_text("dict")["blocks"] if "lines" in b
                 for line in b["lines"] for s in line["spans"]]
        # Removing text only leaves original colored shapes, arrows, and images
        # intact, including the dashed/gray swap-and-pop state in Figure 3.
        page.add_redact_annot(page.rect, fill=False)
        page.apply_redactions(images=0, graphics=0, text=0)
        assert not page.get_text().strip(), source
        for face in FACES:
            page.insert_font(fontname="Libertine" + face,
                             fontfile=str(font_dir / FACES[face]))
        math_x_positions = {}
        for i, span in enumerate(spans[:-1]):
            if "DejaVuMath" not in span["font"] or span["text"] != "ǜ":
                continue
            sub = spans[i + 1]
            if "DejaVuMath" not in sub["font"] or sub["size"] >= span["size"]:
                continue
            subtext = sub["text"].replace("֍", "m")
            subface = "Italic" if subtext == "m" else "Regular"
            xwidth = fonts["Italic"].text_length("x", fontsize=span["size"])
            subwidth = fonts[subface].text_length(subtext, fontsize=sub["size"])
            old_width = sub["bbox"][2] - span["bbox"][0]
            left = span["origin"][0] + (old_width - xwidth - subwidth) / 2
            math_x_positions[i] = left
            math_x_positions[i + 1] = left + xwidth
        for i, span in enumerate(spans):
            text = span["text"]
            if not text.strip():
                continue
            old_name = span["font"]
            is_math = "DejaVuMath" in old_name
            if is_math:
                text = text.replace("ǜ", "x").replace("֍", "m")
            bold = "Bold" in old_name
            italic = "Italic" in old_name or (is_math and text in ("x", "m"))
            face = "BoldItalic" if bold and italic else "Bold" if bold else "Italic" if italic else "Regular"
            assert all(fonts[face].has_glyph(ord(c)) for c in text), (source, text)
            size = span["size"]
            old_width = span["bbox"][2] - span["bbox"][0]
            new_width = fonts[face].text_length(text, fontsize=size)
            # Center labels in their existing footprint, retaining their native
            # font proportions rather than stretching glyphs horizontally.
            x, y = span["origin"]
            x = math_x_positions.get(i, x + (old_width - new_width) / 2)
            color = pymupdf.sRGB_to_pdf(span["color"])
            page.insert_text((x, y), text, fontname="Libertine" + face,
                             fontsize=size, color=color,
                             fill_opacity=span.get("alpha", 255) / 255)
            count += 1
        page.clean_contents(sanitize=True)
        # Redaction retains empty text-state blocks. Drop those before cleaning
        # resources, otherwise an unused Arial font still appears in pdffonts.
        for xref in page.get_contents():
            contents = doc.xref_stream(xref)
            contents = re.sub(
                rb"\bBT\b(.*?)\bET\b",
                lambda m: m.group(0) if re.search(rb"\b(?:Tj|TJ)\b", m.group(1)) else b"",
                contents, flags=re.S,
            )
            doc.update_stream(xref, contents)
        page.clean_contents(sanitize=True)
    assert before_paths == [len(p.get_drawings()) for p in doc], source
    assert before_images == [len(p.get_images(full=True)) for p in doc], source
    doc.subset_fonts()
    doc.save(output, garbage=4, deflate=True, clean=True)
    return {"source": str(source), "output": str(output), "text_spans": count,
            "path_counts": before_paths, "image_counts": before_images}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_dir = find_font_dir(args.font_dir)
    print(json.dumps([restyle(args.source_dir / (name + ".pdf"),
                            args.out_dir / (name + ".pdf"), font_dir)
                      for name in NAMES], indent=2))


if __name__ == "__main__":
    main()
