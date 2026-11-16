"""Build the project page, index.html, from page-prose.md and page-blocks.html.

The prose of the page lives in page-prose.md; everything that is not prose (the head, the viewer
panels and controls, the code listings, the noscript fallback and the script tag) lives in
page-blocks.html as named blocks. The markdown names a block with a placeholder line
``<!-- block: NAME -->`` and the build pastes the block's HTML there verbatim.

The page is a fixed sidebar beside one pane at a time. The sidebar holds a logo (the
``sidebar-logo`` block), a table of contents generated from the sections, their child sections
and their subheadings, and small print (the ``version`` and ``links`` blocks). Everything before
the first ``##`` is the Intro pane (id ``intro``); each section is a pane; a last pane,
Documentation (id ``documentation``), is added from the ``documentation`` block unless the
markdown has a section with that id. The section that places the ``hero-viewer`` block is the
demo: the ``demo-bar`` block (the playback controls), titled with that section's heading, sticks
to the top of the column while the demo or one of its child sections is open. The script nav.js
shows the pane the URL hash names.

The ``markdown`` package is not a dependency of the project, so this module carries a minimal
converter for exactly the subset page-prose.md uses:

- front matter between ``---`` lines: ``title`` and ``description`` fill the head block;
- ``# text``: the heading of the Intro pane;
  ``## text {#id}``: a section, which is a pane with that id; ``## text {#id .sub}`` (id and
  class in either order): a child section, a pane of its own listed under the nearest preceding
  section without ``.sub``; ``### text`` or ``### text {#id}``: a subheading inside a section,
  listed under its pane in the contents, with the given id or one derived from its words;
  ``#### text``: a minor heading inside a section, with an id but not listed in the contents;
- paragraphs (consecutive lines, joined with single spaces); an indented paragraph right after
  a list is a further paragraph of the list's last item; a line ``{.name}`` directly above a
  paragraph gives it that class; by default paragraphs inside a section are ``lead``
  paragraphs and paragraphs in the header and footer have no class; the ``links`` block, named in
  the header, goes to the sidebar instead;
- lists of ``- `` items (continuation lines indented);
- ``> `` lines right after a block placeholder: that block's caption, which fills the block's
  ``{{caption}}`` slot; a blank line between groups of ``> `` lines starts a new paragraph;
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
    python docs/site/build_page.py --prose PATH --out PATH  # build another copy of the markdown
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
HEADING_ATTRS = re.compile(r"^(.+?) \{([^{}]*)\}$")
ATTR_TOKEN = re.compile(r"^([#.])([A-Za-z0-9_-]+)$")
SECTION_CLASSES = {"sub"}  # .sub: a child section of the nearest top-level section
ATTR = re.compile(r"^\{\.([A-Za-z0-9_-]+)\}$")
CAPTION_SLOT = "{{caption}}"
SIDEBAR_LINKS = "links"  # a block the header names, placed in the sidebar instead
DEMO_BLOCK = "hero-viewer"  # the section placing this block is the demo, with the demo bar
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

    def math(m: re.Match) -> str:
        # TeX between $...$ or $$...$$, kept verbatim for the page's math renderer
        body = m.group(0).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return keep(body)

    s = re.sub(r"\$\$.+?\$\$|\$(?=\S)[^$]*?(?<=\S)\$", math, s)
    s = re.sub(r"`([^`]+)`", code, s)
    s = PLACEHOLDER.sub(block, s)
    s = escape_text(s)
    s = s.replace("\\[", "\x01").replace("\\]", "\x02")  # \[ and \] are literal brackets
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    s = s.replace("\x01", "[").replace("\x02", "]")
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
        elif items:
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


def heading_attrs(text: str, allowed: set[str]) -> tuple[str, str | None, set[str]]:
    """Split ``Title {#id .class}`` (tokens in any order) into title, id and classes."""
    m = HEADING_ATTRS.match(text)
    if not m:
        return text, None, set()
    title, ident, classes = m.group(1), None, set()
    for token in m.group(2).split():
        tm = ATTR_TOKEN.match(token)
        if not tm:
            raise PageError(f"cannot read {token!r} in a heading's braces: {text!r}")
        if tm.group(1) == "#":
            if ident is not None:
                raise PageError(f"a heading has two ids: {text!r}")
            ident = tm.group(2)
        elif tm.group(2) not in allowed:
            known = ", ".join(f".{c}" for c in sorted(allowed)) or "none"
            raise PageError(f"unknown class .{tm.group(2)} in {text!r} (known: {known})")
        else:
            classes.add(tm.group(2))
    return title, ident, classes


class Pane:
    """A table-of-contents entry: a pane, its subheadings and its child panes."""

    def __init__(self, ident: str, label: str, parent: str | None = None) -> None:
        self.ident, self.label, self.parent = ident, label, parent
        self.title = label  # the heading's HTML
        self.subs: list[tuple[str, str, list[tuple[str, str]]]] = []  # id, label, minors
        self.children: list[Pane] = []


def toc_items(panes: list[Pane], depth: int) -> list[str]:
    pad = "  " * (2 * depth + 3)
    out = [f"{pad[:-2]}<ul>"]
    for pane in panes:
        out.append(f'{pad}<li><a href="#{pane.ident}" data-pane="{pane.ident}">{pane.label}</a>')
        if pane.subs or pane.children:
            if pane.subs:
                inner = "  " * (2 * depth + 5)
                out.append(f"{inner[:-2]}<ul>")
                for i, t, minors in pane.subs:
                    if not minors:
                        out.append(f'{inner}<li><a href="#{i}">{t}</a></li>')
                        continue
                    deep = inner + "    "
                    out.append(f'{inner}<li><a href="#{i}">{t}</a>')
                    out.append(f"{deep[:-2]}<ul>")
                    out += [f'{deep}<li><a href="#{mi}">{mt}</a></li>' for mi, mt in minors]
                    out.append(f"{deep[:-2]}</ul>")
                    out.append(f"{inner}</li>")
                out.append(f"{inner[:-2]}</ul>")
            if pane.children:
                out += toc_items(pane.children, depth + 1)
            out.append(f"{pad}</li>")
        else:
            out[-1] += "</li>"
    out.append(f"{pad[:-2]}</ul>")
    return out


def sidebar(blocks: dict[str, str], panes: list[Pane]) -> str:
    """The sidebar: logo, table of contents (panes with their subheadings), small print."""
    logo = get_block(blocks, "sidebar-logo").strip()
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
        *toc_items(panes, 0),
        "  </nav>",
        '  <div class="smallprint">',
        f"    <p>Noumenal Labs &middot; {version} &middot; MIT</p>",
        get_block(blocks, SIDEBAR_LINKS),
        "  </div>",
        "  </div>",
        "</aside>",
    ]
    return "\n".join(out)


def demo_bar(blocks: dict[str, str], demo: Pane | None) -> list[str]:
    """The demo's sticky bar, shown over the demo pane and its child panes."""
    if demo is None:
        return []
    title = demo.title
    group = " ".join([demo.ident] + [c.ident for c in demo.children])
    bar = get_block(blocks, "demo-bar").replace("{{title}}", title)
    bar = bar.replace("{{href}}", f"#{demo.ident}").replace("{{panes}}", group)
    return [bar]


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
    panes = [Pane("intro", "Intro")]  # top-level entries; child panes hang under theirs
    current: Pane = panes[0]
    ids = {"intro"}
    demo: Pane | None = None  # the section that places DEMO_BLOCK
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
            text, pane_id, classes = heading_attrs(first[3:], SECTION_CLASSES)
            if pane_id is None:
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
            title = inline(text, blocks)
            if pane_id in ids:
                raise PageError(f"section id {pane_id!r} is used twice")
            ids.add(pane_id)
            if "sub" in classes:
                top = panes[-1]
                if top.ident == "intro":
                    raise PageError(f"a .sub section needs a section before it: {first!r}")
                current = Pane(pane_id, toc_label(title), parent=top.ident)
                top.children.append(current)
                attr = f' data-parent="{top.ident}"'
            else:
                current = Pane(pane_id, toc_label(title))
                panes.append(current)
                attr = ""
            current.title = title
            out += [f'<section class="pane" id="{pane_id}"{attr}>', f"  <h2>{title}</h2>"]
            section_open = True
        elif first.startswith("### "):
            text, sub_id, _ = heading_attrs(first[4:], set())
            title = inline(text, blocks)
            sub_id = sub_id or slug(title)
            if sub_id in ids:
                raise PageError(f"subheading id {sub_id!r} is used twice; give one '{{#id}}'")
            ids.add(sub_id)
            if region == "main":
                current.subs.append((sub_id, toc_label(title), []))
            out.append(f'  <h3 id="{sub_id}">{title}</h3>')
        elif first.startswith("#### "):
            text, minor_id, _ = heading_attrs(first[5:], set())
            title = inline(text, blocks)
            minor_id = minor_id or slug(title)
            if minor_id in ids:
                raise PageError(f"heading id {minor_id!r} is used twice; give one '{{#id}}'")
            ids.add(minor_id)
            out.append(f'  <h4 id="{minor_id}">{title}</h4>')
            out += [f"  <p>{inline(' '.join(chunk[1:]), blocks)}</p>"] if len(chunk) > 1 else []
        elif first.startswith(" ") and out and out[-1].strip() == "</ul>":
            # an indented paragraph after a list: a second paragraph of the list's last item
            end = out.pop()
            item = out.pop()
            assert item.endswith("</li>")
            para = inline(" ".join(line.strip() for line in chunk), blocks)
            out.append(f"{item[:-5]}<p>{para}</p></li>")
            out.append(end)
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
            # the caption: the '> ' lines of this chunk and every following chunk of '> ' lines,
            # a blank line between them starting a new paragraph of the caption
            paragraphs = [chunk[1:]] if chunk[1:] else []
            while i < len(groups) and groups[i][0].startswith(">"):
                paragraphs.append(groups[i])
                i += 1
            caption = [line for para in paragraphs for line in para]
            if (CAPTION_SLOT in html) != bool(caption):
                need = "needs" if caption == [] else "has no slot for"
                raise PageError(f"block {first!r} {need} a caption ('> ' lines after it)")
            if region == "header" and name == SIDEBAR_LINKS:
                continue  # the header links live in the sidebar's small print
            if name == DEMO_BLOCK:
                if region != "main" or demo is not None:
                    raise PageError(f"block {name!r} belongs in one section")
                demo = current
            if caption:
                texts = [
                    inline(" ".join(line[1:].strip() for line in para), blocks)
                    for para in paragraphs
                ]
                text = texts[0] if len(texts) == 1 else "".join(f"<p>{x}</p>" for x in texts)
                html = html.replace(CAPTION_SLOT, text)
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
        panes.append(Pane(DOCS_ID, "Documentation"))
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
        *demo_bar(blocks, demo),
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
    parser.add_argument(
        "--prose", type=Path, default=PROSE, help="the markdown (default: page-prose.md)"
    )
    args = parser.parse_args(argv)
    try:
        page = build(args.prose.read_text(encoding="utf-8"), BLOCKS.read_text(encoding="utf-8"))
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
