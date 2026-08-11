"""Generate an editable Visio-compatible Dual-D framework diagram."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
VSDX_PATH = DOCS / "dual_d_framework_flow.vsdx"
VDX_PATH = DOCS / "dual_d_framework_flow.vdx"
SVG_PATH = DOCS / "dual_d_framework_flow.svg"

NS = "http://schemas.microsoft.com/visio/2003/core"
VSDX_NS = "http://schemas.microsoft.com/office/visio/2011/1/core"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC_NS = "http://purl.org/dc/elements/1.1/"
DCTERMS_NS = "http://purl.org/dc/terms/"
ET.register_namespace("", NS)
ET.register_namespace("r", DOC_REL_NS)
ET.register_namespace("cp", CP_NS)
ET.register_namespace("dc", DC_NS)
ET.register_namespace("dcterms", DCTERMS_NS)


@dataclass(frozen=True)
class Node:
    key: str
    x: float
    y: float
    w: float
    h: float
    label: str
    fill: str
    stroke: str = "#4b5563"


@dataclass(frozen=True)
class Edge:
    start: str
    end: str
    label: str = ""
    style: str = "solid"


def vdx_y(svg_y: float, page_h: float) -> float:
    return page_h - svg_y


def add_text(parent: ET.Element, text: str) -> None:
    text_el = ET.SubElement(parent, f"{{{NS}}}Text")
    text_el.text = text


def add_cell(parent: ET.Element, name: str, value: str) -> None:
    ET.SubElement(parent, f"{{{NS}}}Cell", {"N": name, "V": value})


def hex_to_rgb_int(color: str) -> str:
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return str((r << 16) + (g << 8) + b)


def make_vdx(nodes: list[Node], edges: list[Edge], page_w: float, page_h: float) -> None:
    doc = ET.Element(
        f"{{{NS}}}VisioDocument",
        {
            "xml:space": "preserve",
        },
    )
    props = ET.SubElement(doc, f"{{{NS}}}DocumentProperties")
    ET.SubElement(props, f"{{{NS}}}Title").text = "Dual-D framework flow"
    ET.SubElement(props, f"{{{NS}}}Subject").text = "Dual discriminator multimodal domain adaptation"
    ET.SubElement(props, f"{{{NS}}}Creator").text = "Codex"

    pages = ET.SubElement(doc, f"{{{NS}}}Pages")
    page = ET.SubElement(pages, f"{{{NS}}}Page", {"ID": "0", "NameU": "Page-1", "Name": "Dual-D Framework"})
    page_sheet = ET.SubElement(page, f"{{{NS}}}PageSheet")
    page_props = ET.SubElement(page_sheet, f"{{{NS}}}PageProps")
    ET.SubElement(page_props, f"{{{NS}}}PageWidth").text = f"{page_w:.3f}"
    ET.SubElement(page_props, f"{{{NS}}}PageHeight").text = f"{page_h:.3f}"
    ET.SubElement(page_props, f"{{{NS}}}DrawingScale").text = "1"
    ET.SubElement(page_props, f"{{{NS}}}PageScale").text = "1"
    ET.SubElement(page_props, f"{{{NS}}}DrawingSizeType").text = "0"

    shapes = ET.SubElement(page, f"{{{NS}}}Shapes")
    node_map = {node.key: node for node in nodes}
    shape_id = 1

    for node in nodes:
        shape = ET.SubElement(
            shapes,
            f"{{{NS}}}Shape",
            {"ID": str(shape_id), "NameU": node.key, "Name": node.key, "Type": "Shape"},
        )
        shape_id += 1
        xform = ET.SubElement(shape, f"{{{NS}}}XForm")
        ET.SubElement(xform, f"{{{NS}}}PinX").text = f"{node.x:.3f}"
        ET.SubElement(xform, f"{{{NS}}}PinY").text = f"{vdx_y(node.y, page_h):.3f}"
        ET.SubElement(xform, f"{{{NS}}}Width").text = f"{node.w:.3f}"
        ET.SubElement(xform, f"{{{NS}}}Height").text = f"{node.h:.3f}"
        ET.SubElement(xform, f"{{{NS}}}LocPinX").text = f"{node.w / 2:.3f}"
        ET.SubElement(xform, f"{{{NS}}}LocPinY").text = f"{node.h / 2:.3f}"

        fill = ET.SubElement(shape, f"{{{NS}}}Fill")
        add_cell(fill, "FillForegnd", hex_to_rgb_int(node.fill))
        add_cell(fill, "FillPattern", "1")

        line = ET.SubElement(shape, f"{{{NS}}}Line")
        add_cell(line, "LineColor", hex_to_rgb_int(node.stroke))
        add_cell(line, "LineWeight", "0.012")

        char = ET.SubElement(shape, f"{{{NS}}}Char")
        add_cell(char, "Color", "0")
        add_cell(char, "Size", "9 pt")
        add_cell(char, "Font", "0")

        para = ET.SubElement(shape, f"{{{NS}}}Para")
        add_cell(para, "HorzAlign", "1")

        text_block = ET.SubElement(shape, f"{{{NS}}}TextBlock")
        add_cell(text_block, "VerticalAlign", "1")
        add_cell(text_block, "LeftMargin", "0.08")
        add_cell(text_block, "RightMargin", "0.08")

        geom = ET.SubElement(shape, f"{{{NS}}}Geom", {"IX": "0"})
        move = ET.SubElement(geom, f"{{{NS}}}MoveTo", {"IX": "1"})
        ET.SubElement(move, f"{{{NS}}}X").text = "0"
        ET.SubElement(move, f"{{{NS}}}Y").text = "0"
        for ix, x, y in [
            ("2", node.w, 0),
            ("3", node.w, node.h),
            ("4", 0, node.h),
            ("5", 0, 0),
        ]:
            line_to = ET.SubElement(geom, f"{{{NS}}}LineTo", {"IX": ix})
            ET.SubElement(line_to, f"{{{NS}}}X").text = f"{x:.3f}"
            ET.SubElement(line_to, f"{{{NS}}}Y").text = f"{y:.3f}"

        add_text(shape, node.label)

    for edge in edges:
        src = node_map[edge.start]
        dst = node_map[edge.end]
        start_x = src.x + src.w / 2
        start_y_svg = src.y
        end_x = dst.x - dst.w / 2
        end_y_svg = dst.y
        if abs(end_x - start_x) < 0.05:
            start_x = src.x
            end_x = dst.x
            start_y_svg = src.y - src.h / 2
            end_y_svg = dst.y + dst.h / 2

        begin_y = vdx_y(start_y_svg, page_h)
        end_y = vdx_y(end_y_svg, page_h)
        min_x = min(start_x, end_x)
        min_y = min(begin_y, end_y)
        width = abs(end_x - start_x) or 0.001
        height = abs(end_y - begin_y) or 0.001
        local_start_x = start_x - min_x
        local_start_y = begin_y - min_y
        local_end_x = end_x - min_x
        local_end_y = end_y - min_y

        shape = ET.SubElement(
            shapes,
            f"{{{NS}}}Shape",
            {"ID": str(shape_id), "NameU": f"edge_{edge.start}_{edge.end}", "Type": "Shape"},
        )
        shape_id += 1
        xform = ET.SubElement(shape, f"{{{NS}}}XForm")
        ET.SubElement(xform, f"{{{NS}}}PinX").text = f"{min_x + width / 2:.3f}"
        ET.SubElement(xform, f"{{{NS}}}PinY").text = f"{min_y + height / 2:.3f}"
        ET.SubElement(xform, f"{{{NS}}}Width").text = f"{width:.3f}"
        ET.SubElement(xform, f"{{{NS}}}Height").text = f"{height:.3f}"
        ET.SubElement(xform, f"{{{NS}}}LocPinX").text = f"{width / 2:.3f}"
        ET.SubElement(xform, f"{{{NS}}}LocPinY").text = f"{height / 2:.3f}"

        line = ET.SubElement(shape, f"{{{NS}}}Line")
        add_cell(line, "LineColor", "4210752")
        add_cell(line, "LineWeight", "0.011")
        add_cell(line, "EndArrow", "4")
        if edge.style == "dashed":
            add_cell(line, "LinePattern", "2")

        geom = ET.SubElement(shape, f"{{{NS}}}Geom", {"IX": "0"})
        ET.SubElement(geom, f"{{{NS}}}NoFill").text = "1"
        move = ET.SubElement(geom, f"{{{NS}}}MoveTo", {"IX": "1"})
        ET.SubElement(move, f"{{{NS}}}X").text = f"{local_start_x:.3f}"
        ET.SubElement(move, f"{{{NS}}}Y").text = f"{local_start_y:.3f}"
        line_to = ET.SubElement(geom, f"{{{NS}}}LineTo", {"IX": "2"})
        ET.SubElement(line_to, f"{{{NS}}}X").text = f"{local_end_x:.3f}"
        ET.SubElement(line_to, f"{{{NS}}}Y").text = f"{local_end_y:.3f}"

        if edge.label:
            char = ET.SubElement(shape, f"{{{NS}}}Char")
            add_cell(char, "Size", "7 pt")
            add_cell(shape, "TxtPinX", "Width*0.5")
            add_text(shape, edge.label)

    ET.indent(doc, space="  ")
    tree = ET.ElementTree(doc)
    tree.write(VDX_PATH, encoding="utf-8", xml_declaration=True)


def xml_bytes(root: ET.Element) -> bytes:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def relationship_xml(relationships: list[tuple[str, str, str]]) -> bytes:
    root = ET.Element(f"{{{REL_NS}}}Relationships")
    for rel_id, rel_type, target in relationships:
        ET.SubElement(
            root,
            f"{{{REL_NS}}}Relationship",
            {"Id": rel_id, "Type": rel_type, "Target": target},
        )
    return xml_bytes(root)


def vsdx_cell(parent: ET.Element, name: str, value: str, unit: str | None = None) -> None:
    attrs = {"N": name, "V": value}
    if unit:
        attrs["U"] = unit
    ET.SubElement(parent, f"{{{VSDX_NS}}}Cell", attrs)


def vsdx_geom_row(parent: ET.Element, row_type: str, ix: int, x: float, y: float) -> None:
    row = ET.SubElement(
        parent,
        f"{{{VSDX_NS}}}Row",
        {"T": row_type, "IX": str(ix)},
    )
    vsdx_cell(row, "X", f"{x:.3f}")
    vsdx_cell(row, "Y", f"{y:.3f}")


def add_vsdx_rectangle(shape: ET.Element, node: Node, page_h: float) -> None:
    vsdx_cell(shape, "PinX", f"{node.x:.3f}", "IN")
    vsdx_cell(shape, "PinY", f"{vdx_y(node.y, page_h):.3f}", "IN")
    vsdx_cell(shape, "Width", f"{node.w:.3f}", "IN")
    vsdx_cell(shape, "Height", f"{node.h:.3f}", "IN")
    vsdx_cell(shape, "LocPinX", f"{node.w / 2:.3f}", "IN")
    vsdx_cell(shape, "LocPinY", f"{node.h / 2:.3f}", "IN")
    vsdx_cell(shape, "FillForegnd", node.fill)
    vsdx_cell(shape, "FillPattern", "1")
    vsdx_cell(shape, "LineColor", node.stroke)
    vsdx_cell(shape, "LineWeight", "0.012", "IN")
    vsdx_cell(shape, "VerticalAlign", "1")
    vsdx_cell(shape, "HorzAlign", "1")

    geometry = ET.SubElement(shape, f"{{{VSDX_NS}}}Section", {"N": "Geometry", "IX": "0"})
    vsdx_geom_row(geometry, "MoveTo", 1, 0.0, 0.0)
    vsdx_geom_row(geometry, "LineTo", 2, node.w, 0.0)
    vsdx_geom_row(geometry, "LineTo", 3, node.w, node.h)
    vsdx_geom_row(geometry, "LineTo", 4, 0.0, node.h)
    vsdx_geom_row(geometry, "LineTo", 5, 0.0, 0.0)
    ET.SubElement(shape, f"{{{VSDX_NS}}}Text").text = node.label


def add_vsdx_line(
    shape: ET.Element,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    label: str,
    dashed: bool,
) -> None:
    min_x = min(start_x, end_x)
    min_y = min(start_y, end_y)
    width = abs(end_x - start_x) or 0.001
    height = abs(end_y - start_y) or 0.001
    local_start_x = start_x - min_x
    local_start_y = start_y - min_y
    local_end_x = end_x - min_x
    local_end_y = end_y - min_y

    vsdx_cell(shape, "PinX", f"{min_x + width / 2:.3f}", "IN")
    vsdx_cell(shape, "PinY", f"{min_y + height / 2:.3f}", "IN")
    vsdx_cell(shape, "Width", f"{width:.3f}", "IN")
    vsdx_cell(shape, "Height", f"{height:.3f}", "IN")
    vsdx_cell(shape, "LocPinX", f"{width / 2:.3f}", "IN")
    vsdx_cell(shape, "LocPinY", f"{height / 2:.3f}", "IN")
    vsdx_cell(shape, "LineColor", "#4b5563")
    vsdx_cell(shape, "LineWeight", "0.011", "IN")
    vsdx_cell(shape, "EndArrow", "4")
    if dashed:
        vsdx_cell(shape, "LinePattern", "2")

    geometry = ET.SubElement(shape, f"{{{VSDX_NS}}}Section", {"N": "Geometry", "IX": "0"})
    vsdx_cell(geometry, "NoFill", "1")
    vsdx_geom_row(geometry, "MoveTo", 1, local_start_x, local_start_y)
    vsdx_geom_row(geometry, "LineTo", 2, local_end_x, local_end_y)
    if label:
        ET.SubElement(shape, f"{{{VSDX_NS}}}Text").text = label


def make_vsdx(nodes: list[Node], edges: list[Edge], page_w: float, page_h: float) -> None:
    content_types = ET.Element(f"{{{CT_NS}}}Types")
    for ext, content_type in [
        ("rels", "application/vnd.openxmlformats-package.relationships+xml"),
        ("xml", "application/xml"),
    ]:
        ET.SubElement(content_types, f"{{{CT_NS}}}Default", {"Extension": ext, "ContentType": content_type})
    for part, content_type in [
        ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
        ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml"),
        ("/visio/document.xml", "application/vnd.ms-visio.drawing.main+xml"),
        ("/visio/pages/pages.xml", "application/vnd.ms-visio.pages+xml"),
        ("/visio/pages/page1.xml", "application/vnd.ms-visio.page+xml"),
    ]:
        ET.SubElement(content_types, f"{{{CT_NS}}}Override", {"PartName": part, "ContentType": content_type})

    document = ET.Element(
        f"{{{VSDX_NS}}}VisioDocument",
        {f"{{http://www.w3.org/XML/1998/namespace}}space": "preserve"},
    )
    ET.SubElement(document, f"{{{VSDX_NS}}}DocumentSettings")
    face_names = ET.SubElement(document, f"{{{VSDX_NS}}}FaceNames")
    ET.SubElement(face_names, f"{{{VSDX_NS}}}FaceName", {"ID": "0", "Name": "Microsoft YaHei"})

    pages = ET.Element(
        f"{{{VSDX_NS}}}Pages",
        {f"{{http://www.w3.org/XML/1998/namespace}}space": "preserve"},
    )
    page = ET.SubElement(pages, f"{{{VSDX_NS}}}Page", {"ID": "0", "NameU": "Page-1", "Name": "Dual-D Framework"})
    page_sheet = ET.SubElement(page, f"{{{VSDX_NS}}}PageSheet")
    vsdx_cell(page_sheet, "PageWidth", f"{page_w:.3f}", "IN")
    vsdx_cell(page_sheet, "PageHeight", f"{page_h:.3f}", "IN")
    vsdx_cell(page_sheet, "PageScale", "1", "IN")
    vsdx_cell(page_sheet, "DrawingScale", "1", "IN")
    ET.SubElement(page, f"{{{VSDX_NS}}}Rel", {f"{{{DOC_REL_NS}}}id": "rId1"})

    page_contents = ET.Element(
        f"{{{VSDX_NS}}}PageContents",
        {f"{{http://www.w3.org/XML/1998/namespace}}space": "preserve"},
    )
    shapes = ET.SubElement(page_contents, f"{{{VSDX_NS}}}Shapes")
    node_map = {node.key: node for node in nodes}
    shape_id = 1
    for node in nodes:
        shape = ET.SubElement(
            shapes,
            f"{{{VSDX_NS}}}Shape",
            {"ID": str(shape_id), "NameU": node.key, "Name": node.key, "Type": "Shape"},
        )
        shape_id += 1
        add_vsdx_rectangle(shape, node, page_h)

    for edge in edges:
        src = node_map[edge.start]
        dst = node_map[edge.end]
        start_x = src.x + src.w / 2
        start_y_svg = src.y
        end_x = dst.x - dst.w / 2
        end_y_svg = dst.y
        if abs(end_x - start_x) < 0.05:
            start_x = src.x
            end_x = dst.x
            start_y_svg = src.y - src.h / 2
            end_y_svg = dst.y + dst.h / 2
        shape = ET.SubElement(
            shapes,
            f"{{{VSDX_NS}}}Shape",
            {
                "ID": str(shape_id),
                "NameU": f"edge_{edge.start}_{edge.end}",
                "Name": f"edge_{edge.start}_{edge.end}",
                "Type": "Shape",
            },
        )
        shape_id += 1
        add_vsdx_line(
            shape,
            start_x,
            vdx_y(start_y_svg, page_h),
            end_x,
            vdx_y(end_y_svg, page_h),
            edge.label,
            edge.style == "dashed",
        )

    core = ET.Element(f"{{{CP_NS}}}coreProperties")
    ET.SubElement(core, f"{{{DC_NS}}}title").text = "Dual-D framework flow"
    ET.SubElement(core, f"{{{DC_NS}}}creator").text = "Codex"
    ET.SubElement(core, f"{{{CP_NS}}}keywords").text = "Dual-D; multimodal domain adaptation; Visio"

    app = ET.Element(
        "Properties",
        {"xmlns": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"},
    )
    ET.SubElement(app, "Application").text = "Microsoft Visio"
    ET.SubElement(app, "DocSecurity").text = "0"
    ET.SubElement(app, "ScaleCrop").text = "false"

    with zipfile.ZipFile(VSDX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", xml_bytes(content_types))
        zf.writestr(
            "_rels/.rels",
            relationship_xml(
                [
                    ("rId1", "http://schemas.microsoft.com/visio/2010/relationships/document", "visio/document.xml"),
                    ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
                    ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
                ]
            ),
        )
        zf.writestr(
            "visio/_rels/document.xml.rels",
            relationship_xml(
                [
                    ("rId1", "http://schemas.microsoft.com/visio/2010/relationships/pages", "pages/pages.xml"),
                ]
            ),
        )
        zf.writestr(
            "visio/pages/_rels/pages.xml.rels",
            relationship_xml(
                [
                    ("rId1", "http://schemas.microsoft.com/visio/2010/relationships/page", "page1.xml"),
                ]
            ),
        )
        zf.writestr("visio/document.xml", xml_bytes(document))
        zf.writestr("visio/pages/pages.xml", xml_bytes(pages))
        zf.writestr("visio/pages/page1.xml", xml_bytes(page_contents))
        zf.writestr("docProps/core.xml", xml_bytes(core))
        zf.writestr("docProps/app.xml", xml_bytes(app))


def svg_rect(node: Node, scale: float) -> str:
    x = (node.x - node.w / 2) * scale
    y = (node.y - node.h / 2) * scale
    w = node.w * scale
    h = node.h * scale
    lines = escape(node.label).split("\n")
    text_y = y + h / 2 - (len(lines) - 1) * 8
    text = "".join(
        f'<tspan x="{x + w / 2:.1f}" dy="{0 if idx == 0 else 16}">{line}</tspan>'
        for idx, line in enumerate(lines)
    )
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="8" fill="{node.fill}" stroke="{node.stroke}" stroke-width="1.4"/>\n'
        f'<text x="{x + w / 2:.1f}" y="{text_y:.1f}" text-anchor="middle" '
        f'dominant-baseline="middle" font-size="13" fill="#111827">{text}</text>'
    )


def make_svg(nodes: list[Node], edges: list[Edge], page_w: float, page_h: float) -> None:
    scale = 86.0
    node_map = {node.key: node for node in nodes}
    width = page_w * scale
    height = page_h * scale
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#4b5563"/>',
        "</marker>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="22" font-weight="600" fill="#111827">Dual-D 多模态域适应整体流程图</text>',
    ]
    for edge in edges:
        src = node_map[edge.start]
        dst = node_map[edge.end]
        x1 = (src.x + src.w / 2) * scale
        y1 = src.y * scale
        x2 = (dst.x - dst.w / 2) * scale
        y2 = dst.y * scale
        if abs(x2 - x1) < 4:
            x1 = src.x * scale
            y1 = (src.y + src.h / 2) * scale
            x2 = dst.x * scale
            y2 = (dst.y - dst.h / 2) * scale
        dash = ' stroke-dasharray="7 5"' if edge.style == "dashed" else ""
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#4b5563" stroke-width="1.6"{dash} marker-end="url(#arrow)"/>'
        )
        if edge.label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 - 6
            parts.append(
                f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" font-size="11" '
                f'fill="#374151">{escape(edge.label)}</text>'
            )
    for node in nodes:
        parts.append(svg_rect(node, scale))
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    page_w = 15.8
    page_h = 10.6
    nodes = [
        Node("source_input", 1.35, 2.0, 1.8, 0.78, "源域输入\n晴天 VIS/IR + label", "#dbeafe"),
        Node("target_input", 1.35, 4.15, 1.8, 0.78, "目标域输入\n黑天 VIS/IR + label", "#fee2e2"),
        Node("encoder_s", 3.65, 2.0, 1.95, 0.78, "VIS/IR 编码器\nResNet + IR CNN", "#e0f2fe"),
        Node("encoder_t", 3.65, 4.15, 1.95, 0.78, "VIS/IR 编码器\n共享 VIS 权重", "#e0f2fe"),
        Node("tal", 6.0, 3.05, 2.15, 0.92, "TAL 张量对齐\n跨模态/跨域投影", "#dcfce7"),
        Node("fs", 8.25, 2.0, 1.55, 0.7, "源域融合特征\nFs", "#f0fdf4"),
        Node("ft", 8.25, 4.15, 1.55, 0.7, "目标域融合特征\nFt", "#fff7ed"),
        Node("classifier", 10.55, 1.0, 1.75, 0.72, "分类器 C\n类别 logits", "#ede9fe"),
        Node("cls_loss", 13.05, 1.0, 1.75, 0.72, "分类损失\n源域 + 目标域", "#f5f3ff"),
        Node("g_t2s", 10.55, 3.38, 1.9, 0.72, "G_t2s\n目标 -> 源风格", "#fef3c7"),
        Node("source_like", 13.0, 3.38, 1.75, 0.72, "source-like\nFts", "#fefce8"),
        Node("d_source", 15.0, 3.38, 1.72, 0.72, "主判别器 Ds\n是否像真实源域", "#dbeafe"),
        Node("g_s2t", 10.55, 5.05, 1.9, 0.72, "G_s2t\n源 -> 目标风格", "#ffedd5"),
        Node("target_like", 13.0, 5.05, 1.75, 0.72, "target-like\nFst", "#fff7ed"),
        Node("d_target", 15.0, 5.05, 1.72, 0.72, "辅助判别器 Dt\n是否像真实目标域", "#fee2e2"),
        Node("cycle", 12.0, 6.55, 2.05, 0.8, "Cycle 一致性\n双向重建约束", "#ecfccb"),
        Node("identity", 9.3, 6.55, 1.9, 0.8, "Identity 保持\n避免无意义改写", "#ecfccb"),
        Node("contrast", 14.65, 6.55, 2.05, 0.8, "对比 + 分类反馈\n保持类别语义", "#fce7f3"),
        Node("total_loss", 12.0, 8.0, 2.1, 0.82, "总训练目标\n联合优化", "#e5e7eb"),
        Node("inference", 5.0, 9.35, 6.35, 0.82, "推理路径：目标 VIS/IR -> 编码器 -> TAL target 投影 -> G_t2s -> 分类器 -> 预测", "#f8fafc"),
    ]
    edges = [
        Edge("source_input", "encoder_s"),
        Edge("target_input", "encoder_t"),
        Edge("encoder_s", "tal"),
        Edge("encoder_t", "tal"),
        Edge("tal", "fs"),
        Edge("tal", "ft"),
        Edge("fs", "classifier"),
        Edge("ft", "classifier"),
        Edge("classifier", "cls_loss"),
        Edge("ft", "g_t2s"),
        Edge("g_t2s", "source_like"),
        Edge("source_like", "d_source", "对抗"),
        Edge("fs", "g_s2t"),
        Edge("g_s2t", "target_like"),
        Edge("target_like", "d_target", "对抗"),
        Edge("source_like", "cycle", "重建 Ft", "dashed"),
        Edge("target_like", "cycle", "重建 Fs", "dashed"),
        Edge("fs", "identity", style="dashed"),
        Edge("ft", "identity", style="dashed"),
        Edge("source_like", "contrast", style="dashed"),
        Edge("target_like", "contrast", style="dashed"),
        Edge("cls_loss", "total_loss"),
        Edge("d_source", "total_loss"),
        Edge("d_target", "total_loss"),
        Edge("cycle", "total_loss"),
        Edge("identity", "total_loss"),
        Edge("contrast", "total_loss"),
        Edge("ft", "inference", "eval: source_like", "dashed"),
    ]
    # Visio is strict about the packaged .vsdx structure. The editable .vsdx
    # is generated by scripts/generate_dual_d_visio_com.ps1 through Visio's
    # native object model; this helper only maintains VDX/SVG companion files.
    make_vdx(nodes, edges, page_w, page_h)
    make_svg(nodes, edges, page_w, page_h)
    ET.parse(VDX_PATH)
    ET.parse(SVG_PATH)
    print(f"Wrote {VDX_PATH}")
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    main()
