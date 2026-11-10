"""Build the launch page, index.html, from page-prose.md and page-blocks.html.

The prose of the page lives in page-prose.md; everything that is not prose (the head, the viewer
panels and controls, the code listings, the noscript fallback and the script tag) lives in
page-blocks.html as named blocks. The markdown names a block with a placeholder line
``<!-- block: NAME -->`` and the build pastes the block's HTML there verbatim.

The page is a fixed sidebar beside one pane at a time. The sidebar holds the logo (the
``title-image`` block), a table of contents generated from the sections and their subheadings,
and small print (the ``version`` and ``links`` blocks). Everything before the first ``##`` is the
Intro pane (id ``intro``); each section is a pane; a last pane, Documentation (id
``documentation``), is added from the ``documentation`` block unless the markdown has a section
with that id. The script nav.js shows the pane the URL hash names.

The ``markdown`` package is not a dependency of the project, so this module carries a minimal
converter for exactly the subset page-prose.md uses:

- front matter between ``---`` lines: ``title`` and ``description`` fill the head block;
- ``# text``: the page title in the Intro pane (visually hidden; the sidebar shows the logo);
  ``## text {#id}``: a section, which is a pane with that id; ``### text`` or
  ``### text {#id}``: a subheading inside a section, listed under its pane in the contents, with
  the given id or one derived from its words;
- paragraphs (consecutive lines, joined with single spaces); a line ``{.name}`` directly above a
  paragraph gives it that class; by default paragraphs inside a section are ``lead``
  paragraphs and paragraphs in the header and footer have no class; the ``links`` block, named in
  the header, goes to the sidebar instead;
- lists of ``- `` items (continuation lines indented);
- ``> `` lines right after a block placeholder: that block's caption, which fills the block's
  ``{{caption}}`` slot;
- the markers ``<!-- tldr -->`` (opens the header's TLDR, plain prose) and ``<!-- footer -->``
  (ends the sections; the paragraphs after it form the footer under every pane);
- a comment opened by a line ``<!--`` and closed by a line ``-->`` is ignored;
- inline: ``**strong**``, ``*em*``, ```code```, ``[text](url)``, and a block placeholder inside
  a paragraph, which is pasted inline. A non-ASCII character is written as its named HTML entity
  (``'`` stays a plain apostrophe, while the typographic one becomes ``&rsquo;``).

The output is a pure function of the two source files. Usage, from the repository root or from
docs/site::

    python docs/site/build_page.py            # write index.html
    python docs/site/build_page.py --check    # exit 1 if index.html differs from the build
    python docs/site/build_page.py --out PATH # write the build elsewhere
"""

from __future__ import annotations

import argparse
import html.entities
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROSE = HERE / "page-prose.md"
BLOCKS = HERE / "page-blocks.html"
PAGE = HERE / "index.html"

PLACEHOLDER = re.compile(r"<!-- block: ([A-Za-z0-9_-]+) -->")
BLOCK_OPEN = re.compile(r"^<!-- block: ([A-Za-z0-9_-]+) -->$")
BLOCK_CLOSE = "<!-- /block -->"
SECTION = re.compile(r"^## (.+?) \{#([A-Za-z0-9_-]+)\}$")
ATTR = re.compile(r"^\{\.([A-Za-z0-9_-]+)\}$")
SUBHEADING_ID = re.compile(r"^(.+?) \{#([A-Za-z0-9_-]+)\}$")
CAPTION_SLOT = "{{caption}}"
SIDEBAR_LINKS = "links"  # a block the header names, placed in the sidebar instead
DOCS_ID = "documentation"  # the synthetic last pane, unless the markdown has a section with this id
# set before the layout is painted, so that panes are hidden only when the script can show them
JS_FLAG = '<script>document.documentElement.classList.add("js");</script>'


class PageError(Exception):
    """A malformed source file."""


def read_blocks(text: str) -> dict[str, str]:
    """Named blocks of page-blocks.html; lines outside a block are ignored."""
    blocks: dict[str, str] = {}
    name: str | None = None
    body: list[str] = []
    for line in text.split("\n"):
        if name is None:
            m = BLOCK_OPEN.match(line)
            if m:
                name, body = m.group(1), []
                if name in blocks:
                    raise PageError(f"block {name!r} defined twice")
        elif line == BLOCK_CLOSE:
            blocks[name] = "\n".join(body)
            name = None
        else:
            body.append(line)
    if name is not None:
        raise PageError(f"block {name!r} is not closed")
    return blocks


def entity(ch: str) -> str:
    name = html.entities.codepoint2name.get(ord(ch))
    return f"&{name};" if name else f"&#{ord(ch)};"


def escape_text(s: str) -> str:
    """Escape markup characters in prose, keeping entities an editor wrote by hand."""
    s = re.sub(r"&(?![A-Za-z]+;|#[0-9]+;)", "&amp;", s)
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    return "".join(c if ord(c) < 128 else entity(c) for c in s)


def inline(s: str, blocks: dict[str, str]) -> str:
    """Convert one paragraph's markdown to HTML."""
    saved: list[str] = []

    def keep(fragment: str) -> str:
        saved.append(fragment)
        return f"\x00{len(saved) - 1}\x00"

    def code(m: re.Match) -> str:
        body = m.group(1).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return keep(f"<code>{escape_text(body)}</code>")

    def block(m: re.Match) -> str:
        return keep(get_block(blocks, m.group(1)))

    s = re.sub(r"`([^`]+)`", code, s)
    s = PLACEHOLDER.sub(block, s)
    s = escape_text(s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    return re.sub("\x00([0-9]+)\x00", lambda m: saved[int(m.group(1))], s)


def get_block(blocks: dict[str, str], name: str) -> str:
    if name not in blocks:
        raise PageError(f"no block named {name!r} in {BLOCKS.name}")
    return blocks[name]


def split_front_matter(text: str) -> tuple[dict[str, str], list[str]]:
    lines = text.split("\n")
    if not lines or lines[0] != "---":
        raise PageError("page-prose.md must start with front matter between --- lines")
    meta: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line == "---":
            return meta, lines[i + 1 :]
        key, sep, value = line.partition(":")
        if not sep:
            raise PageError(f"front matter line without a key: {line!r}")
        meta[key.strip()] = value.strip()
    raise PageError("front matter is not closed")


def chunks(lines: list[str]) -> list[list[str]]:
    """Blank-line separated groups of lines, with ``<!--`` ... ``-->`` comments dropped."""
    out: list[list[str]] = []
    cur: list[str] = []
    in_comment = False
    for line in lines:
        if in_comment:
            in_comment = line.strip() != "-->"
            continue
        if line.strip() == "<!--":
            in_comment = True
            continue
        if line.strip():
            cur.append(line.rstrip())
        elif cur:
            out.append(cur)
            cur = []
    if in_comment:
        raise PageError("a comment opened with <!-- is not closed with -->")
    if cur:
        out.append(cur)
    return out


def list_items(chunk: list[str]) -> list[str]:
    items: list[str] = []
    for line in chunk:
        if line.startswith("- "):
            items.append(line[2:].strip())
        elif line.startswith(" ") and items:
            items[-1] += " " + line.strip()
        else:
            raise PageError(f"unexpected line in a list: {line!r}")
    return items


def slug(text: str) -> str:
    """An id for a subheading without an explicit one: lowercase words joined by hyphens."""
    text = re.sub(r"<[^>]+>|&[A-Za-z0-9#]+;", "", text).lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "sub"


def toc_label(fragment: str) -> str:
    """A heading's HTML without its tags, for the table of contents."""
    return re.sub(r"<[^>]+>", "", fragment)


def sidebar(blocks: dict[str, str], panes: list[tuple[str, str, list[tuple[str, str]]]]) -> str:
    """The sidebar: logo, table of contents (panes with their subheadings), small print."""
    logo = get_block(blocks, "title-image").strip()
    version = get_block(blocks, "version").strip().replace('id="version"', 'class="version"')
    out = [
        '<aside class="sidebar" id="sidebar">',
        '  <div class="sidebar-top">',
        f'    <a class="brand" href="#intro">{logo}</a>',
        '    <button class="menu" id="menu" type="button" aria-expanded="false"'
        ' aria-controls="sidebar-panel" hidden>Menu</button>',
        "  </div>",
        '  <div class="sidebar-panel" id="sidebar-panel">',
        '  <nav class="toc" aria-label="Contents">',
        "    <ul>",
    ]
    for pane_id, label, subs in panes:
        item = f'      <li><a href="#{pane_id}" data-pane="{pane_id}">{label}</a>'
        if subs:
            out.append(item)
            out.append("        <ul>")
            out += [f'          <li><a href="#{i}">{t}</a></li>' for i, t in subs]
            out.append("        </ul>")
            out.append("      </li>")
        else:
            out.append(item + "</li>")
    out += [
        "    </ul>",
        "  </nav>",
        '  <div class="smallprint">',
        f"    <p>Noumenal Labs &middot; {version} &middot; MIT</p>",
        get_block(blocks, SIDEBAR_LINKS),
        "  </div>",
        "  </div>",
        "</aside>",
    ]
    return "\n".join(out)


def build(prose: str, blocks_text: str) -> str:
    blocks = read_blocks(blocks_text)
    meta, body = split_front_matter(prose)
    for key in ("title", "description"):
        if key not in meta:
            raise PageError(f"front matter needs {key!r}")
    head = get_block(blocks, "head")
    head = head.replace("{{title}}", escape_text(meta["title"]))
    head = head.replace("{{description}}", escape_text(meta["description"]).replace('"', "&quot;"))

    out = ['<section class="pane" id="intro" aria-label="Intro">']
    # (pane id, ToC label, [(subheading id, label)]) in page order
    panes: list[tuple[str, str, list[tuple[str, str]]]] = [("intro", "Intro", [])]
    ids = {"intro"}
    region = "header"  # header, main, footer
    in_tldr = False
    section_open = False
    footer: list[str] = []
    groups = chunks(body)
    i = 0
    while i < len(groups):
        chunk = groups[i]
        i += 1
        first = chunk[0]
        indent = "    " if in_tldr else "  "
        cls = None
        m = ATTR.match(first)
        if m:
            cls, chunk = m.group(1), chunk[1:]
            if not chunk:
                raise PageError(f"attribute line {first!r} must sit directly above a paragraph")
            first = chunk[0]

        if first == "<!-- tldr -->":
            if region != "header" or in_tldr:
                raise PageError("<!-- tldr --> belongs in the header, once")
            out.append('  <div class="tldr">')
            in_tldr = True
        elif first == "<!-- footer -->":
            if region != "main":
                raise PageError("<!-- footer --> must follow the sections")
            out += ["</section>", ""]
            region = "footer"
        elif first.startswith("# "):
            if region != "header":
                raise PageError("# heading belongs in the header")
            out.append(f'  <h1 class="intro-title">{inline(first[2:], blocks)}</h1>')
        elif first.startswith("## "):
            sm = SECTION.match(first)
            if not sm:
                raise PageError(
                    f"a section heading needs an id, as in '## Title {{#id}}': {first!r}"
                )
            if region == "header":
                if in_tldr:
                    out.append("  </div>")
                    in_tldr = False
                out += ["</section>", ""]
                region = "main"
            elif region == "footer":
                raise PageError("no sections after <!-- footer -->")
            if section_open:
                out += ["</section>", ""]
            title, pane_id = inline(sm.group(1), blocks), sm.group(2)
            if pane_id in ids:
                raise PageError(f"section id {pane_id!r} is used twice")
            ids.add(pane_id)
            panes.append((pane_id, toc_label(title), []))
            out += [f'<section class="pane" id="{pane_id}">', f"  <h2>{title}</h2>"]
            section_open = True
        elif first.startswith("### "):
            text = first[4:]
            hm = SUBHEADING_ID.match(text)
            title = inline(hm.group(1) if hm else text, blocks)
            sub_id = hm.group(2) if hm else slug(title)
            if sub_id in ids:
                raise PageError(f"subheading id {sub_id!r} is used twice; give one '{{#id}}'")
            ids.add(sub_id)
            if region == "main":
                panes[-1][2].append((sub_id, toc_label(title)))
            out.append(f'  <h3 id="{sub_id}">{title}</h3>')
        elif first.startswith("- "):
            out.append(f"{indent}<ul>")
            out += [f"{indent}  <li>{inline(t, blocks)}</li>" for t in list_items(chunk)]
            out.append(f"{indent}</ul>")
        elif first.startswith(">"):
            raise PageError(f"a caption must follow a block with a caption slot: {first!r}")
        elif BLOCK_OPEN.match(first):
            if not all(line.startswith(">") for line in chunk[1:]):
                raise PageError(f"{first!r} must be followed by a blank line or '> ' caption lines")
            name = BLOCK_OPEN.match(first).group(1)
            html = get_block(blocks, name)
            caption = chunk[1:]
            if not caption and i < len(groups) and groups[i][0].startswith(">"):
                caption = groups[i]
                i += 1
            if (CAPTION_SLOT in html) != bool(caption):
                need = "needs" if caption == [] else "has no slot for"
                raise PageError(f"block {first!r} {need} a caption ('> ' lines after it)")
            if region == "header" and name == SIDEBAR_LINKS:
                continue  # the header links live in the sidebar's small print
            if caption:
                text = " ".join(line[1:].strip() for line in caption)
                html = html.replace(CAPTION_SLOT, inline(text, blocks))
            out.append(html)
        else:
            text = inline(" ".join(line.strip() for line in chunk), blocks)
            if region == "footer":
                footer.append(f"<p>{text}</p>")
                continue
            if cls is None and region == "main":
                cls = "lead"
            attr = f' class="{cls}"' if cls else ""
            out.append(f"{indent}<p{attr}>{text}</p>")
    if region != "footer":
        raise PageError("page-prose.md must end with <!-- footer --> and the footer text")
    if DOCS_ID not in ids:
        panes.append((DOCS_ID, "Documentation", []))
        out += [
            f'<section class="pane" id="{DOCS_ID}">',
            "  <h2>Documentation</h2>",
            get_block(blocks, "documentation"),
            "</section>",
            "",
        ]
    page = [
        head,
        JS_FLAG,
        sidebar(blocks, panes),
        "",
        '<div class="column">',
        '<main id="main">',
        *out,
        "</main>",
        f'<footer class="pagefoot">{"".join(footer)}</footer>',
        "</div>",
        get_block(blocks, "tail"),
    ]
    return "\n".join(page) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out", type=Path, default=PAGE, help="where to write (default: index.html)"
    )
    parser.add_argument(
        "--check", action="store_true", help="compare with index.html, write nothing"
    )
    args = parser.parse_args(argv)
    try:
        page = build(PROSE.read_text(encoding="utf-8"), BLOCKS.read_text(encoding="utf-8"))
    except PageError as err:
        print(f"build_page: {err}", file=sys.stderr)
        return 2
    if args.check:
        same = PAGE.read_text(encoding="utf-8") == page
        print("index.html is up to date" if same else "index.html differs from the build")
        return 0 if same else 1
    args.out.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
