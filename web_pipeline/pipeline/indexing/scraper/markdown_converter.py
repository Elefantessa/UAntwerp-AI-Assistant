"""
Markdown Converter
==================

Convert cleaned HTML to Markdown with special handling for UAntwerp pages.
"""

import re
from typing import List

from bs4 import BeautifulSoup, NavigableString, Tag


def _table_to_markdown(table: Tag) -> str:
    """Convert HTML table to Markdown pipe table."""
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        row = [c.get_text(" ", strip=True) for c in cells]
        if row:
            rows.append(row)
    if not rows:
        return ""
    widths = [max(len(cell) for cell in col) for col in zip(*rows)]

    def fmt_row(r):
        return "| " + " | ".join(cell.ljust(w) for cell, w in zip(r, widths)) + " |"

    header = rows[0]
    separator = ["-" * w for w in widths]
    body = rows[1:] if len(rows) > 1 else []
    parts = [fmt_row(header), "| " + " | ".join(separator) + " |"]
    parts += [fmt_row(r) for r in body]
    return "\n".join(parts)


def _squash_blank_lines(lines: List[str]) -> List[str]:
    """Remove consecutive blank lines."""
    out, prev_blank = [], False
    for ln in lines:
        blank = (ln.strip() == "")
        if blank and prev_blank:
            continue
        out.append(ln)
        prev_blank = blank
    return out


def normalize_headings(md: str) -> str:
    """Normalize heading levels to start from H1."""
    levels = []
    for line in md.splitlines():
        m = re.match(r'^(#{1,6})\s', line)
        if m:
            levels.append(len(m.group(1)))
    if not levels:
        return md
    min_level = min(levels)
    if min_level == 1:
        return md
    shift = min_level - 1

    def _fix(line: str):
        m = re.match(r'^(#{1,6})\s(.*)$', line)
        if not m:
            return line
        old = len(m.group(1))
        new = max(1, old - shift)
        return f"{'#'*new} {m.group(2)}"

    return "\n".join(_fix(l) for l in md.splitlines())


def remove_personalpage_urls_from_markdown(md: str) -> str:
    """Remove PersonalPage URLs while keeping names."""
    # [Name](/PersonalPage/en/06908) -> Name
    md = re.sub(r'\[([^\]]+)\]\(/PersonalPage/en/\d+\)', r'\1', md)
    # bare (/PersonalPage/en/06908) -> ''
    md = re.sub(r'\(/PersonalPage/en/\d+\)', '', md)
    # [Name](/PersonalPage/en/06908)Name -> Name
    md = re.sub(r'\[([^\]]+)\]\(/PersonalPage/en/\d+\)\1', r'\1', md)
    return md


def collapse_duplicate_person_names(md: str) -> str:
    """
    Removes adjacent duplicates for patterns like 'Firstname LastnameFirstname Lastname'
    (2-3 capitalized tokens). Safe heuristic focused on person-name repetitions only.
    """
    name = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'\.-]+"
    pat2 = re.compile(rf"\b({name}\s+{name})\1\b")
    pat3 = re.compile(rf"\b({name}\s+{name}\s+{name})\1\b")
    md = pat2.sub(r"\1", md)
    md = pat3.sub(r"\1", md)
    return md


def html_to_markdown(html: str) -> str:
    """
    Convert cleaned HTML to Markdown with special handling for UAntwerp study-programme pages.
    - Specialized extraction for course units (course name + spec label/value bullets).
    - Avoid duplicating link text and drop dynamic '?id=...' links (keep label only).
    - Convert tables to pipe tables before traversal.
    - Fall back to a generic walker for the remaining content.
    """
    soup = BeautifulSoup(html, "lxml")

    # --- 1) Pre: convert any <table> to textual Markdown so it's traversable as text
    for table in soup.find_all("table"):
        md_tbl = _table_to_markdown(table)
        table.replace_with(soup.new_string(md_tbl))

    # Helper: inline text with awareness of links
    def render_inline(node: Tag) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        if not isinstance(node, Tag):
            return ""
        if node.name == "a" and node.has_attr("href"):
            label = node.get_text(" ", strip=True)
            href = (node["href"] or "").strip()
            # Drop dynamic study-programme fiche links like '?id=2025-...&lang=en'
            if href.startswith("?") or "/PersonalPage/en/" in href:
                return label
            return f"[{label}]({href})"
        parts = []
        for child in node.children:
            parts.append(render_inline(child))
        return "".join(parts)

    # Helper: get clean text from a node (single line)
    def inline_text(node: Tag) -> str:
        return " ".join(render_inline(node).split())

    # --- 2) Specialized extraction for study-programme blocks
    spec_lines = []
    course_units = soup.select("section.courseUnit")
    if course_units:
        for unit in course_units:
            # Unit heading (e.g., "Compulsory courses", "Elective courses")
            h_unit = unit.find(["h3", "h4"], class_="heading")
            unit_title = inline_text(h_unit) if h_unit else ""
            if unit_title:
                spec_lines.append(f"### {unit_title}")
                spec_lines.append("")

            # Optional blurb paragraph under the unit
            for txtblock in unit.select(":scope > .textblock, :scope > .main > .textblock"):
                txt = inline_text(txtblock).strip()
                if txt:
                    spec_lines.append(txt)
                    spec_lines.append("")

            # Courses inside this unit
            for course in unit.select("section.course"):
                # Course name (usually in h5.heading > a)
                h_course = course.find(["h5", "h4"], class_="heading")
                course_name = inline_text(h_course) if h_course else "Course"
                if course_name:
                    spec_lines.append(f"#### {course_name}")

                # Specs are in section.fiche div.spec (label/value)
                fiche = course.select_one("section.fiche")
                if fiche:
                    specs = []
                    for spec in fiche.select("div.spec"):
                        label_node = spec.find("div", class_="label")
                        value_node = spec.find("div", class_="value")
                        if not label_node or not value_node:
                            continue
                        label = inline_text(label_node).rstrip(":").strip()

                        # If value has list items, join by comma; else plain inline text
                        lis = value_node.find_all("li")
                        if lis:
                            vals = [inline_text(li) for li in lis if inline_text(li)]
                            value = ", ".join(v for v in vals if v)
                        else:
                            value = inline_text(value_node)

                        value = value.replace("  ", " ").strip()
                        if label and value:
                            specs.append(f"- **{label}:** {value}")

                    if specs:
                        spec_lines.extend(specs)

                spec_lines.append("")

        # Remove courseUnit blocks from soup so the generic walker doesn't duplicate them
        for unit in course_units:
            unit.decompose()

    # --- 3) Generic walker for the remainder (intro text, headings, etc.)
    generic_lines = []

    def walk(block: Tag, level: int = 0):
        if isinstance(block, NavigableString) or not isinstance(block, Tag):
            return
        name = (block.name or "").lower()

        if name in {"[document]", "html", "body", "section", "div", "details", "summary", "header", "main", "article"}:
            for ch in block.children:
                walk(ch, level)
            return

        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            lvl = int(name[1])
            txt = block.get_text(" ", strip=True)
            if txt:
                generic_lines.append(f"{'#'*lvl} {txt}")
                generic_lines.append("")
            return

        if name == "p":
            txt = render_inline(block).strip()
            if txt:
                generic_lines.append(txt)
                generic_lines.append("")
            return

        if name in {"ul", "ol"}:
            for li in block.find_all("li", recursive=False):
                li_txt = render_inline(li).strip()
                if li_txt:
                    generic_lines.append(("  " * level) + "- " + li_txt.split("\n")[0])
                # handle nested lists
                for sub in li.find_all(["ul", "ol"], recursive=False):
                    walk(sub, level + 1)
            generic_lines.append("")
            return

        # Fallback
        txt = render_inline(block).strip()
        if txt:
            generic_lines.append(txt)
            generic_lines.append("")

    container = soup if soup.body is None else soup.body
    walk(container)

    # --- 4) Compose result: generic content first (intro + headings), then the course units
    all_lines = generic_lines + ([""] if generic_lines and spec_lines else []) + spec_lines

    # --- 5) Post-processing & hygiene
    md = "\n".join(_squash_blank_lines(all_lines)).strip()

    # Remove ZW spaces and normalize headings; scrub PersonalPage links; collapse duplicated names
    md = md.replace("\u200b", "").replace("\u200e", "").replace("\u200f", "")
    md = normalize_headings(md)
    md = remove_personalpage_urls_from_markdown(md)
    md = collapse_duplicate_person_names(md)

    return (md.rstrip() + "\n") if md else ""


class MarkdownConverter:
    """
    High-level interface for HTML to Markdown conversion.
    """

    def convert(self, html: str) -> str:
        """Convert HTML to Markdown."""
        return html_to_markdown(html)

    def normalize_headings(self, md: str) -> str:
        """Normalize heading levels."""
        return normalize_headings(md)
