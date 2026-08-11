param(
    [string]$OutputPath = "",
    [string]$PreviewPath = "",
    [switch]$IncludeLegacyModuleCPage
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path (Resolve-Path ".").Path "docs\dual_d_framework_flow.vsdx"
}

$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$OutputDir = [System.IO.Path]::GetDirectoryName($OutputPath)
if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}
if (-not [string]::IsNullOrWhiteSpace($PreviewPath)) {
    $PreviewPath = [System.IO.Path]::GetFullPath($PreviewPath)
    $PreviewDir = [System.IO.Path]::GetDirectoryName($PreviewPath)
    if (-not (Test-Path -LiteralPath $PreviewDir)) {
        New-Item -ItemType Directory -Path $PreviewDir | Out-Null
    }
}

function Rgb-Formula {
    param([string]$Hex)
    $clean = $Hex.TrimStart("#")
    $r = [Convert]::ToInt32($clean.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($clean.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($clean.Substring(4, 2), 16)
    return "RGB($r,$g,$b)"
}

function Set-CellFormula {
    param($Shape, [string]$CellName, [string]$Formula)
    try {
        $Shape.CellsU($CellName).FormulaU = $Formula
    } catch {
        # Keep generation robust across Visio versions.
    }
}

function Convert-UnicodeEscapes {
    param([string]$Text)
    $decoded = [regex]::Replace($Text, "\\u([0-9a-fA-F]{4})", {
        param($m)
        return [string][char][Convert]::ToInt32($m.Groups[1].Value, 16)
    })
    return $decoded.Replace("\n", "`n")
}

function New-Node {
    param(
        [string]$Key,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [string]$Label,
        [string]$Fill,
        [string]$Role = "module"
    )
    return @{
        key = $Key
        x = $X
        y = $Y
        w = $W
        h = $H
        label = $Label
        fill = $Fill
        role = $Role
    }
}

function New-Panel {
    param(
        [string]$Key,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [string]$Label
    )
    return @{
        key = $Key
        x = $X
        y = $Y
        w = $W
        h = $H
        label = $Label
    }
}

function New-Edge {
    param(
        [string]$Start,
        [string]$End,
        [string]$Label = "",
        [bool]$Dashed = $false,
        [string]$From = "E",
        [string]$To = "W",
        [string]$Color = "#1f4e79",
        [bool]$Arrow = $true
    )
    return @{
        start = $Start
        end = $End
        label = $Label
        dashed = $Dashed
        from = $From
        to = $To
        color = $Color
        arrow = $Arrow
    }
}

function New-Segment {
    param(
        [double]$X1,
        [double]$Y1,
        [double]$X2,
        [double]$Y2,
        [string]$Color = "#64748b",
        [bool]$Dashed = $true,
        [bool]$Arrow = $false
    )
    return @{
        x1 = $X1
        y1 = $Y1
        x2 = $X2
        y2 = $Y2
        color = $Color
        dashed = $Dashed
        arrow = $Arrow
    }
}

function Get-AnchorPoint {
    param($Shape, [string]$Anchor)
    $pinX = $Shape.CellsU("PinX").ResultIU
    $pinY = $Shape.CellsU("PinY").ResultIU
    $halfW = $Shape.CellsU("Width").ResultIU / 2.0
    $halfH = $Shape.CellsU("Height").ResultIU / 2.0
    switch ($Anchor) {
        "W" { return @(($pinX - $halfW), $pinY) }
        "E" { return @(($pinX + $halfW), $pinY) }
        "N" { return @($pinX, ($pinY + $halfH)) }
        "S" { return @($pinX, ($pinY - $halfH)) }
        default { return @($pinX, $pinY) }
    }
}

function Add-DetailPanel {
    param($Page, [double]$PageHeight, $Panel)
    $centerY = $PageHeight - [double]$Panel.y
    $shape = $Page.DrawRectangle(
        ([double]$Panel.x - [double]$Panel.w / 2.0),
        ($centerY - [double]$Panel.h / 2.0),
        ([double]$Panel.x + [double]$Panel.w / 2.0),
        ($centerY + [double]$Panel.h / 2.0)
    )
    $shape.NameU = [string]$Panel.key
    $shape.Text = Convert-UnicodeEscapes ([string]$Panel.label)
    Set-CellFormula $shape "FillForegnd" "RGB(250,251,252)"
    Set-CellFormula $shape "FillPattern" "1"
    Set-CellFormula $shape "LineColor" "RGB(190,199,210)"
    Set-CellFormula $shape "LineWeight" "0.65 pt"
    Set-CellFormula $shape "Rounding" "0.025 in"
    Set-CellFormula $shape "Char.Size" "8.5 pt"
    Set-CellFormula $shape "Char.Style" "1"
    Set-CellFormula $shape "Para.HorzAlign" "0"
    Set-CellFormula $shape "VerticalAlign" "0"
    Set-CellFormula $shape "LeftMargin" "0.12 in"
    Set-CellFormula $shape "TopMargin" "0.08 in"
    return $shape
}

function Add-DetailNode {
    param($Page, [double]$PageHeight, $Node)
    $centerY = $PageHeight - [double]$Node.y
    $shape = $Page.DrawRectangle(
        ([double]$Node.x - [double]$Node.w / 2.0),
        ($centerY - [double]$Node.h / 2.0),
        ([double]$Node.x + [double]$Node.w / 2.0),
        ($centerY + [double]$Node.h / 2.0)
    )
    $shape.NameU = [string]$Node.key
    $shape.Text = Convert-UnicodeEscapes ([string]$Node.label)
    Set-CellFormula $shape "FillForegnd" (Rgb-Formula ([string]$Node.fill))
    Set-CellFormula $shape "FillPattern" "1"
    Set-CellFormula $shape "LineColor" "RGB(71,85,105)"
    Set-CellFormula $shape "LineWeight" "0.85 pt"
    Set-CellFormula $shape "Rounding" "0.035 in"
    Set-CellFormula $shape "Char.Size" "7.8 pt"
    Set-CellFormula $shape "Para.HorzAlign" "1"
    Set-CellFormula $shape "VerticalAlign" "1"
    switch ([string]$Node.role) {
        "title" {
            Set-CellFormula $shape "FillPattern" "0"
            Set-CellFormula $shape "LinePattern" "0"
            Set-CellFormula $shape "Char.Size" "13 pt"
            Set-CellFormula $shape "Char.Style" "1"
            Set-CellFormula $shape "Para.HorzAlign" "0"
        }
        "subtitle" {
            Set-CellFormula $shape "FillPattern" "0"
            Set-CellFormula $shape "LinePattern" "0"
            Set-CellFormula $shape "Char.Size" "8 pt"
            Set-CellFormula $shape "Para.HorzAlign" "0"
        }
        "annotation" {
            Set-CellFormula $shape "FillPattern" "0"
            Set-CellFormula $shape "LinePattern" "0"
            Set-CellFormula $shape "Char.Size" "6.8 pt"
        }
        "lane" {
            Set-CellFormula $shape "LineColor" "RGB(148,163,184)"
            Set-CellFormula $shape "LineWeight" "0.65 pt"
            Set-CellFormula $shape "Char.Size" "7.2 pt"
            Set-CellFormula $shape "Char.Style" "1"
        }
        "loss" {
            Set-CellFormula $shape "LineColor" "RGB(126,139,156)"
            Set-CellFormula $shape "Char.Size" "7.2 pt"
        }
        "whw_detail" {
            Set-CellFormula $shape "LineColor" "RGB(177,145,70)"
            Set-CellFormula $shape "LineWeight" "0.75 pt"
            Set-CellFormula $shape "Rounding" "0.025 in"
            Set-CellFormula $shape "Char.Size" "6.5 pt"
            Set-CellFormula $shape "Para.HorzAlign" "0"
            Set-CellFormula $shape "LeftMargin" "0.10 in"
            Set-CellFormula $shape "RightMargin" "0.08 in"
        }
        "formula" {
            Set-CellFormula $shape "LineColor" "RGB(150,160,174)"
            Set-CellFormula $shape "LineWeight" "0.65 pt"
            Set-CellFormula $shape "Char.Size" "6.6 pt"
            Set-CellFormula $shape "Para.HorzAlign" "0"
            Set-CellFormula $shape "LeftMargin" "0.08 in"
        }
    }
    return $shape
}

function Add-DetailEdge {
    param($Page, $ShapeMap, $Edge)
    $src = $ShapeMap[[string]$Edge.start]
    $dst = $ShapeMap[[string]$Edge.end]
    $startPoint = Get-AnchorPoint $src ([string]$Edge.from)
    $endPoint = Get-AnchorPoint $dst ([string]$Edge.to)
    $line = $Page.DrawLine($startPoint[0], $startPoint[1], $endPoint[0], $endPoint[1])
    $line.NameU = "detail_edge_" + [string]$Edge.start + "_" + [string]$Edge.end
    Set-CellFormula $line "LineColor" (Rgb-Formula ([string]$Edge.color))
    Set-CellFormula $line "LineWeight" "0.9 pt"
    if ([bool]$Edge.dashed) {
        Set-CellFormula $line "LinePattern" "2"
    }
    if ([bool]$Edge.arrow) {
        Set-CellFormula $line "EndArrow" "4"
    } else {
        Set-CellFormula $line "EndArrow" "0"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$Edge.label)) {
        $line.Text = Convert-UnicodeEscapes ([string]$Edge.label)
        Set-CellFormula $line "Char.Size" "6 pt"
    }
    return $line
}

function Add-DetailSegment {
    param($Page, [double]$PageHeight, $Segment, [int]$Index)
    $line = $Page.DrawLine(
        [double]$Segment.x1,
        ($PageHeight - [double]$Segment.y1),
        [double]$Segment.x2,
        ($PageHeight - [double]$Segment.y2)
    )
    $line.NameU = "detail_segment_" + $Index
    Set-CellFormula $line "LineColor" (Rgb-Formula ([string]$Segment.color))
    Set-CellFormula $line "LineWeight" "0.85 pt"
    if ([bool]$Segment.dashed) {
        Set-CellFormula $line "LinePattern" "2"
    }
    if ([bool]$Segment.arrow) {
        Set-CellFormula $line "EndArrow" "4"
    } else {
        Set-CellFormula $line "EndArrow" "0"
    }
    return $line
}

$pageWidth = 25.0
$pageHeight = 15.55

$panels = @(
    New-Panel "panel_representation" 7.40 2.10 14.10 2.10 "A  \u591a\u6a21\u6001\u8868\u793a\u4e0e\u5f20\u91cf\u5bf9\u9f50"
    New-Panel "panel_translation" 14.05 4.75 10.10 2.70 "B  \u53cc\u5411\u57df\u7ffb\u8bd1\u4e0e\u53cc\u5224\u522b\u5668"
    New-Panel "panel_constraints" 14.05 9.00 12.20 5.70 "C  \u8bed\u4e49\u4fdd\u6301\u4e0e\u7c7b\u522b\u611f\u77e5\u53cd\u9988\uff08\u96c6\u6210\u5f0f\u7ea6\u675f\u8868\uff09"
    New-Panel "panel_objective" 22.55 4.95 4.35 7.80 "D  \u8054\u5408\u8bad\u7ec3\u76ee\u6807"
    New-Panel "panel_inference" 12.45 13.15 15.65 1.45 "E  \u63a8\u7406\u8def\u5f84\uff08\u4ec5\u4fdd\u7559\u5fc5\u8981\u6a21\u5757\uff09"
)

$nodes = @(
    New-Node "title" 8.35 0.30 15.80 0.38 "Dual-D \u591a\u6a21\u6001\u8de8\u57df\u8bc6\u522b\u6846\u67b6" "#ffffff" "title"
    New-Node "subtitle" 8.35 0.70 15.80 0.28 "\u591a\u6a21\u6001\u5f20\u91cf\u5bf9\u9f50 \u00b7 \u53cc\u5411\u57df\u7ffb\u8bd1 \u00b7 \u7c7b\u522b\u611f\u77e5\u5bf9\u6bd4\u53cd\u9988" "#ffffff" "subtitle"
    New-Node "legend" 19.00 0.70 9.20 0.28 "\u56fe\u4f8b\uff5c\u84dd\u8272\u5b9e\u7ebf\uff1a\u7279\u5f81\u524d\u5411\u6d41    \u6a59\u8272\u865a\u7ebf\uff1a\u53cd\u5411\u4f20\u64ad    \u7eff\u8272\u5b9e\u7ebf\uff1a\u635f\u5931\u6c47\u603b    \u7070\u8272\u7ec6\u7ebf\uff1a\u6a21\u5757\u8bf4\u660e\u5173\u8054" "#ffffff" "legend"

    New-Node "input_pair" 1.75 2.20 2.20 0.98 "\u6210\u5bf9\u6279\u6b21\u8f93\u5165\n\u6e90\u57df\uff08\u6674\u5929\uff09VIS/IR + y_s\n\u76ee\u6807\u57df\uff08\u9ed1\u5929\uff09VIS/IR + y_t" "#dcecff"
    New-Node "encoder" 4.20 2.20 1.90 0.78 "\u5171\u4eab VIS/IR \u7f16\u7801\u5668\nE_v\uff0cE_ir" "#dcecff"
    New-Node "tal" 6.65 2.20 2.10 0.82 "TAL \u591a\u6a21\u6001\u5f20\u91cf\u5bf9\u9f50\n\u8de8\u6a21\u6001 + \u8de8\u57df\u6295\u5f71" "#d9f3ee"
    New-Node "feature_pair" 9.10 2.20 1.90 0.82 "\u878d\u5408\u7279\u5f81\nF_s\uff08\u6e90\u57df\uff09/ F_t\uff08\u76ee\u6807\u57df\uff09" "#e5f5ec"
    New-Node "classifier" 11.45 2.20 1.70 0.72 "\u4efb\u52a1\u5206\u7c7b\u5668 C\n\u7c7b\u522b logits" "#ece9fb"
    New-Node "cls_loss" 13.55 2.20 1.70 0.72 "L_cls\n\u76d1\u7763\u5206\u7c7b\u635f\u5931" "#eef1f5" "loss"

    New-Node "g_t2s" 10.55 4.20 1.75 0.70 "G_t2s\nF_t \u2192 \u6e90\u57df\u98ce\u683c" "#fff0d9"
    New-Node "source_like" 12.75 4.20 1.80 0.70 "source-like \u7279\u5f81\nF_ts = G_t2s(F_t)" "#fff8ea"
    New-Node "d_source" 15.05 4.20 1.95 0.74 "\u4e3b\u5224\u522b\u5668 D_s\n\u771f F_s / \u5047 F_ts" "#fde8e7"
    New-Node "adv_s_loss" 17.35 4.20 1.70 0.68 "L_adv^s\n\u76ee\u6807 \u2192 \u6e90\u57df\u5bf9\u9f50" "#eef1f5" "loss"

    New-Node "g_s2t" 10.55 5.35 1.75 0.70 "G_s2t\nF_s \u2192 \u76ee\u6807\u57df\u98ce\u683c" "#fff0d9"
    New-Node "target_like" 12.75 5.35 1.80 0.70 "target-like \u7279\u5f81\nF_st = G_s2t(F_s)" "#fff8ea"
    New-Node "d_target" 15.05 5.35 1.95 0.74 "\u8f85\u52a9\u5224\u522b\u5668 D_t\n\u771f F_t / \u5047 F_st" "#fde8e7"
    New-Node "adv_t_loss" 17.35 5.35 1.70 0.68 "L_adv^t\n\u6e90\u57df \u2192 \u76ee\u6807\u5bf9\u9f50" "#eef1f5" "loss"

    New-Node "module_b_io" 14.05 3.60 7.40 0.25 "B \u8f93\u5165\uff1aF_s\uff0cF_t    \u2192    B \u8f93\u51fa\uff1aF_ts\uff0cF_st\uff0cL_adv^s\uff0cL_adv^t" "#ffffff" "annotation"
    New-Node "constraint_inputs" 14.05 6.48 11.30 0.34 "B \u2192 C  \u5171\u4eab\u8f93\u5165\u603b\u7ebf\uff1a{F_s, F_t, F_ts, F_st, y_s, y_t, C, G_t2s, G_s2t}" "#e7eef7" "annotation"

    New-Node "c1_what" 9.10 7.35 2.05 0.92 "C1  WHAT\n\u57df\u5185\u7279\u5f81 F_s,F_t\n\u8fdb\u5165\u5bf9\u5e94\u8f93\u51fa\u57df\u751f\u6210\u5668" "#dcecff" "c_step"
    New-Node "c1_how" 14.00 7.35 7.25 0.92 "C1  HOW\uff5cIdentity \u6052\u7b49\u6620\u5c04\nF_s^id = G_t2s(F_s),   F_t^id = G_s2t(F_t)\nL_id = ||F_s^id-F_s||_1 + ||F_t^id-F_t||_1" "#e9f5e6" "formula"
    New-Node "c1_why" 18.85 7.35 2.20 0.92 "C1  WHY / OUTPUT\n\u6291\u5236\u65e0\u8c13\u57df\u98ce\u683c\u6539\u5199\n\u4fdd\u62a4\u8239\u4f53\u5f31\u8f6e\u5ed3 \u2192 L_id" "#eee9fb" "c_step"

    New-Node "c2_what" 9.10 8.55 2.05 1.02 "C2  WHAT\n\u7ffb\u8bd1\u7279\u5f81 F_ts,F_st\n\u518d\u6620\u5c04\u56de\u539f\u59cb\u57df" "#dcecff" "c_step"
    New-Node "c2_how" 14.00 8.55 7.25 1.02 "C2  HOW\uff5cCycle \u53cc\u5411\u95ed\u73af\nF_t_hat = G_s2t(F_ts),   F_s_hat = G_t2s(F_st)\nL_cyc = ||F_s_hat-F_s||_1 + ||F_t_hat-F_t||_1" "#e9f5e6" "formula"
    New-Node "c2_why" 18.85 8.55 2.20 1.02 "C2  WHY / OUTPUT\n\u8981\u6c42\u7ffb\u8bd1\u53ef\u9006\n\u907f\u514d\u8239\u53ea\u5c3a\u5bf8/\u7ed3\u6784\u4e22\u5931 \u2192 L_cyc" "#eee9fb" "c_step"

    New-Node "c3_what" 9.10 9.95 2.05 1.48 "C3  WHAT\n\u771f\u5b9e/\u7ffb\u8bd1\u7279\u5f81\nF_s,F_t,F_ts,F_st\n\u4e0e\u6807\u7b7e y_s,y_t" "#dcecff" "c_step"
    New-Node "c3_how" 14.00 9.95 7.25 1.48 "C3  HOW\uff5c\u914d\u5bf9 + \u7c7b\u522b\u539f\u578b\u5bf9\u6bd4\nP_s^c = mean_{i:y_s_i=c}(F_s_i),   P_t^c = mean_{i:y_t_i=c}(F_t_i)\nL_con = 1/2[PCE(F_ts,F_s;y) + PCE(F_st,F_t;y)]\nL_proto = 1/2[CE(sim(F_ts,P_s)/tau,y_t) + CE(sim(F_st,P_t)/tau,y_s)]" "#eee9fb" "formula"
    New-Node "c3_why" 18.85 9.95 2.20 1.48 "C3  WHY / OUTPUT\n\u4fdd\u6301\u6210\u5bf9\u5bf9\u5e94\uff1b\u540c\u7c7b\u805a\u5408\n\u5f02\u7c7b\u5206\u79bb\uff0c\u9632\u6b62\u5c0f\u76ee\u6807\u7c7b\u522b\u6df7\u53e0\n\u2192 L_con, L_proto" "#eee9fb" "c_step"

    New-Node "c4_what" 9.10 11.25 2.05 0.92 "C4  WHAT\n\u7ffb\u8bd1\u7279\u5f81 F_ts,F_st\n\u9001\u56de\u4efb\u52a1\u5206\u7c7b\u5668 C" "#dcecff" "c_step"
    New-Node "c4_how" 14.00 11.25 7.25 0.92 "C4  HOW\uff5c\u751f\u6210\u7279\u5f81\u5206\u7c7b\u53cd\u9988\nz_ts = C(F_ts),   z_st = C(F_st)\nL_gcls = CE(z_ts,y_t) + CE(z_st,y_s)" "#e9f5e6" "formula"
    New-Node "c4_why" 18.85 11.25 2.20 0.92 "C4  WHY / OUTPUT\n\u57df\u5bf9\u9f50\u540e\u4ecd\u7136\u53ef\u5206\u7c7b\n\u76f4\u63a5\u4fdd\u62a4\u4efb\u52a1\u8bed\u4e49 \u2192 L_gcls" "#eee9fb" "c_step"

    New-Node "c_output" 20.03 9.35 0.20 5.05 "" "#e8f3e8" "loss_bus"
    New-Node "objective_inputs" 22.55 8.25 3.70 0.45 "\u63a5\u6536 C \u6a21\u5757\u635f\u5931\n\u52a0\u5165 L_total \u8054\u5408\u4f18\u5316" "#e8f3e8" "objective_detail"
    New-Node "backprop_label" 13.00 12.12 9.60 0.24 "\u6a59\u8272\u865a\u7ebf\u7bad\u5934\uff1aL_total \u7684\u68af\u5ea6\u56de\u4f20\u81f3 E_v/E_ir\u3001TAL\u3001G_t2s/G_s2t\u3001C\u4e0e D_s/D_t" "#ffffff" "annotation"

    New-Node "note_A" 3.55 4.45 6.70 1.85 "A \u6a21\u5757\uff1a\u591a\u6a21\u6001\u8868\u793a\u4e0e\u5f20\u91cf\u5bf9\u9f50\nWHAT\uff5c\u878d\u5408 VIS/IR\uff0c\u751f\u6210\u5bf9\u9f50\u7684\u6e90/\u76ee\u6807\u57df\u7279\u5f81 F_s\u3001F_t\u3002\nHOW \uff5c\u5171\u4eab VIS/IR \u7f16\u7801\u5668\u63d0\u53d6\u5916\u89c2\u4e0e\u70ed\u4fe1\u606f\uff1bTAL \u5bf9\u8de8\u6a21\u6001/\u8de8\u57df\u5f20\u91cf\u6295\u5f71\u3001\u5bf9\u9f50\u5e76\u878d\u5408\u3002\nWHY \uff5c\u96e8\u96fe\u3001\u4f4e\u7167\u548c\u6d77\u9762\u53cd\u5149\u4f1a\u4e25\u91cd\u7834\u574f VIS\uff0c\u800c IR \u8f83\u7a33\u5b9a\uff1b\u8054\u5408\u5bf9\u9f50\u53ef\u964d\u4f4e\u5929\u6c14/\u4f20\u611f\u5668\u57df\u504f\u79fb\uff0c\u540c\u65f6\u4fdd\u7559\u8239\u4f53\u7ed3\u6784\u3002" "#fff9e8" "whw"
    New-Node "note_B" 3.55 6.55 6.70 1.85 "B \u6a21\u5757\uff1a\u53cc\u5411\u57df\u7ffb\u8bd1\u4e0e\u53cc\u5224\u522b\u5668\nWHAT\uff5c\u5c06\u6076\u52a3\u5929\u6c14\u76ee\u6807\u7279\u5f81\u8f6c\u4e3a source-like\uff0c\u540c\u65f6\u5b66\u4e60\u53cd\u5411 target-like \u6620\u5c04\u3002\nHOW \uff5c\u6b8b\u5dee\u5f0f G_t2s/G_s2t \u6267\u884c\u53cc\u5411\u7279\u5f81\u7ffb\u8bd1\uff1bD_s/D_t \u5206\u522b\u5224\u522b\u6e90/\u76ee\u6807\u57df\uff0c\u5f62\u6210\u65b9\u5411\u611f\u77e5\u5bf9\u6297\u5b66\u4e60\u3002\nWHY \uff5c\u9ed1\u5929\u3001\u96e8\u96fe\u4e0e\u5f3a\u6d77\u6742\u6ce2\u9020\u6210\u663e\u8457\u57df\u504f\u79fb\uff1b\u5355\u5411\u9002\u914d\u6613\u584c\u7f29\u6216\u53ea\u9a97\u8fc7\u5224\u522b\u5668\uff0c\u53cc\u5411\u5206\u5e03\u5efa\u6a21\u66f4\u7a33\u5b9a\u3002" "#fff9e8" "whw"
    New-Node "note_C" 3.55 8.65 6.70 1.85 "C \u6a21\u5757\uff1a\u8bed\u4e49\u4fdd\u6301\u4e0e\u7c7b\u522b\u611f\u77e5\u53cd\u9988\nWHAT\uff5c\u5728\u57df\u5bf9\u9f50\u65f6\u4fdd\u6301\u7c7b\u522b\u8bed\u4e49\u3001\u76ee\u6807\u7ed3\u6784\u548c\u7ffb\u8bd1\u53ef\u9006\u6027\u3002\nHOW \uff5cIdentity \u9632\u6b62\u65e0\u610f\u4e49\u6539\u5199\uff1bCycle \u95ed\u73af\u91cd\u5efa\uff1b\u914d\u5bf9/\u539f\u578b\u5bf9\u6bd4\u62c9\u8fd1\u540c\u7c7b\u3001\u63a8\u8fdc\u5f02\u7c7b\uff1b\u751f\u6210\u5206\u7c7b\u53cd\u9988\u4fdd\u6301\u53ef\u5206\u6027\u3002\nWHY \uff5c\u6d77\u4e0a\u5c0f\u76ee\u6807\u7c7b\u95f4\u5dee\u5f02\u5c0f\u3001\u6d77\u6d6a\u80cc\u666f\u5f3a\uff1b\u7eaf\u5bf9\u6297\u5bf9\u9f50\u53ef\u80fd\u6df7\u5408\u7c7b\u522b\u5e76\u62b9\u6389\u5f31\u76ee\u6807\u7279\u5f81\u3002" "#fff9e8" "whw"
    New-Node "note_D" 22.55 9.95 4.35 1.90 "D \u6a21\u5757\uff1a\u8054\u5408\u8bad\u7ec3\u76ee\u6807\nWHAT\uff5c\u7edf\u4e00\u4efb\u52a1\u3001\u5bf9\u9f50\u3001\u5bf9\u6297\u3001\u91cd\u5efa\u4e0e\u7c7b\u522b\u7ea6\u675f\u3002\nHOW \uff5c\u52a0\u6743 L_total \u8054\u5408\u4f18\u5316\uff1b\u5224\u522b\u5668/\u751f\u6210\u5668\u4ea4\u66ff\u66f4\u65b0\uff1b\u5bf9\u6297\u6743\u91cd\u9884\u70ed\u4e0e\u6e10\u589e\u3002\nWHY \uff5c\u5e73\u8861\u8bc6\u522b\u51c6\u786e\u7387\u4e0e\u57df\u4e0d\u53d8\u6027\uff1b\u5206\u9636\u6bb5\u5bf9\u6297\u53ef\u907f\u514d\u5224\u522b\u5668\u8fc7\u5f3a\u5bfc\u81f4\u635f\u5931\u9707\u8361\u548c\u68af\u5ea6\u4e0d\u7a33\u3002" "#fff9e8" "whw_compact"
    New-Node "note_E" 12.45 14.77 15.65 1.35 "E \u6a21\u5757\uff1a\u6076\u52a3\u5929\u6c14\u63a8\u7406\u8def\u5f84\nWHAT\uff5c\u5bf9\u76ee\u6807\u57df VIS/IR \u6837\u672c\u8f93\u51fa\u6d77\u4e0a\u76ee\u6807\u7c7b\u522b\u9884\u6d4b\u3002\nHOW \uff5c\u76ee\u6807 VIS/IR \u2192 \u5171\u4eab\u7f16\u7801\u5668 \u2192 TAL \u2192 G_t2s \u2192 C\uff1b\u90e8\u7f72\u65f6\u79fb\u9664\u6e90\u57df\u8f93\u5165\u3001D_s/D_t\u3001G_s2t \u548c\u8bad\u7ec3\u635f\u5931\u3002\nWHY \uff5c\u4ec5\u4fdd\u7559\u9c81\u68d2\u8bc6\u522b\u4e3b\u8def\uff0c\u51cf\u5c11\u6d77\u4e0a\u76d1\u63a7\u7ad9\u6216\u65e0\u4eba\u5e73\u53f0\u7684\u5728\u7ebf\u90e8\u7f72\u5f00\u9500\u3002" "#fff9e8" "whw_wide"

    New-Node "total_loss" 22.55 1.95 3.75 0.75 "\u603b\u635f\u5931\uff08\u8054\u5408\u4f18\u5316\uff09\nL_total = L_cls + \u03bb_TAL L_TAL + \u03bb_s L_adv^s + \u03bb_t L_adv^t\n+ \u03bb_cyc L_cyc + \u03bb_id L_id + \u03bb_con L_con + \u03bb_proto L_proto + \u03bb_gcls L_gcls" "#e4e9ef" "objective"
    New-Node "loss_detail_cls" 22.55 2.70 3.75 0.50 "L_cls = CE(C(F_s),y_s) + CE(C(F_t),y_t)\n\u76f4\u63a5\u5206\u7c7b\uff1a\u4fdd\u7559\u4efb\u52a1\u8bed\u4e49" "#f7f8fa" "objective_detail"
    New-Node "loss_detail_tal" 22.55 3.25 3.75 0.50 "L_TAL = ||A_s - A_t||_2^2\n\u5f20\u91cf\u5bf9\u9f50\uff1a\u51cf\u5c11\u6a21\u6001\u4e0e\u57df\u5dee\u5f02" "#f7f8fa" "objective_detail"
    New-Node "loss_detail_adv" 22.55 3.95 3.75 0.78 "L_adv^s = CE(D_s(F_ts),1)\uff1bL_adv^t = CE(D_t(F_st),1)\nL_D = 1/2(L_D^s + L_D^t)\uff0c\u771f\u5b9e=1\uff0c\u751f\u6210=0\n\u5bf9\u6297\u5b66\u4e60\uff1a\u5b9e\u73b0\u53cc\u5411\u57df\u5206\u5e03\u5bf9\u9f50" "#f7f8fa" "objective_detail"
    New-Node "loss_detail_cyc" 22.55 4.67 3.75 0.52 "L_cyc = ||F_s_hat-F_s||_1 + ||F_t_hat-F_t||_1\n\u5faa\u73af\u4e00\u81f4\uff1a\u7ea6\u675f\u53cc\u5411\u95ed\u73af\u91cd\u5efa" "#f7f8fa" "objective_detail"
    New-Node "loss_detail_id" 22.55 5.30 3.75 0.62 "L_id = ||G_t2s(F_s)-F_s||_1\n      + ||G_s2t(F_t)-F_t||_1\n\u8eab\u4efd\u4fdd\u6301\uff1a\u907f\u514d\u65e0\u610f\u4e49\u98ce\u683c\u6539\u5199" "#f7f8fa" "objective_detail"
    New-Node "loss_detail_con" 22.55 5.94 3.75 0.52 "L_con = 1/2[PCE(F_ts,F_s) + PCE(F_st,F_t)]\n\u914d\u5bf9\u5bf9\u6bd4\uff1a\u4fdd\u6301\u5c40\u90e8\u7ed3\u6784" "#f7f8fa" "objective_detail"
    New-Node "loss_detail_proto" 22.55 6.66 3.75 0.78 "L_proto = 1/2[CE(sim(F_ts,P_s^c)/\u03c4,y_t)\n          + CE(sim(F_st,P_t^c)/\u03c4,y_s)]\n\u539f\u578b\u5bf9\u6bd4\uff1a\u62c9\u8fd1\u540c\u7c7b\uff0c\u63a8\u8fdc\u5f02\u7c7b" "#f7f8fa" "objective_detail"
    New-Node "loss_detail_gcls" 22.55 7.43 3.75 0.58 "L_gcls = CE(C(F_ts),y_t) + CE(C(F_st),y_s)\n\u5206\u7c7b\u53cd\u9988\uff1a\u4fdd\u8bc1\u7ffb\u8bd1\u540e\u4ecd\u53ef\u5206\u7c7b" "#f7f8fa" "objective_detail"

    New-Node "infer_input" 5.45 13.15 1.75 0.60 "\u76ee\u6807 VIS/IR" "#dcecff"
    New-Node "infer_encoder" 8.05 13.15 1.75 0.60 "\u5171\u4eab\u7f16\u7801\u5668" "#dcecff"
    New-Node "infer_tal" 10.65 13.15 1.75 0.60 "TAL \u76ee\u6807\u6295\u5f71" "#d9f3ee"
    New-Node "infer_translate" 13.25 13.15 1.75 0.60 "G_t2s \u6e90\u57df\u5316" "#fff0d9"
    New-Node "infer_classifier" 15.85 13.15 1.75 0.60 "\u5206\u7c7b\u5668 C" "#ece9fb"
    New-Node "infer_prediction" 18.45 13.15 1.75 0.60 "\u7c7b\u522b\u9884\u6d4b" "#e5f5ec"
)

$edges = @(
    New-Edge "input_pair" "encoder" "" $false "E" "W" "#1f4e79"
    New-Edge "encoder" "tal" "" $false "E" "W" "#1f4e79"
    New-Edge "tal" "feature_pair" "" $false "E" "W" "#1f4e79"
    New-Edge "feature_pair" "classifier" "" $false "E" "W" "#1f4e79"
    New-Edge "classifier" "cls_loss" "" $false "E" "W" "#1f4e79"

    New-Edge "feature_pair" "module_b_io" "" $false "S" "W" "#1f4e79"
    New-Edge "feature_pair" "g_t2s" "" $false "S" "W" "#1f4e79"
    New-Edge "feature_pair" "g_s2t" "" $false "S" "W" "#1f4e79"
    New-Edge "g_t2s" "source_like" "" $false "E" "W" "#1f4e79"
    New-Edge "source_like" "d_source" "" $false "E" "W" "#1f4e79"
    New-Edge "d_source" "adv_s_loss" "" $false "E" "W" "#2e7d32"
    New-Edge "g_s2t" "target_like" "" $false "E" "W" "#1f4e79"
    New-Edge "target_like" "d_target" "" $false "E" "W" "#1f4e79"
    New-Edge "d_target" "adv_t_loss" "" $false "E" "W" "#2e7d32"

    New-Edge "c1_what" "c1_how" "" $false "E" "W" "#1f4e79"
    New-Edge "c1_how" "c1_why" "" $false "E" "W" "#2e7d32"
    New-Edge "c2_what" "c2_how" "" $false "E" "W" "#1f4e79"
    New-Edge "c2_how" "c2_why" "" $false "E" "W" "#2e7d32"
    New-Edge "c3_what" "c3_how" "" $false "E" "W" "#1f4e79"
    New-Edge "c3_how" "c3_why" "" $false "E" "W" "#2e7d32"
    New-Edge "c4_what" "c4_how" "" $false "E" "W" "#1f4e79"
    New-Edge "c4_how" "c4_why" "" $false "E" "W" "#2e7d32"

    New-Edge "cls_loss" "classifier" "" $true "W" "E" "#c55a11"
    New-Edge "adv_s_loss" "d_source" "" $true "W" "E" "#c55a11"
    New-Edge "adv_t_loss" "d_target" "" $true "W" "E" "#c55a11"
    New-Edge "c_output" "objective_inputs" "" $false "E" "W" "#2e7d32"

    New-Edge "infer_input" "infer_encoder"
    New-Edge "infer_encoder" "infer_tal"
    New-Edge "infer_tal" "infer_translate"
    New-Edge "infer_translate" "infer_classifier"
    New-Edge "infer_classifier" "infer_prediction"
)

$constraintSegments = @(
    # A/B output bundle enters C from the right edge without crossing B's two lanes.
    New-Segment 17.75 3.60 20.00 3.60 "#1f4e79" $false $false
    New-Segment 20.00 3.60 20.00 6.48 "#1f4e79" $false $false
    New-Segment 20.00 6.48 19.70 6.48 "#1f4e79" $false $true

    # One shared feature bus feeds four compact WHAT -> HOW -> WHY rows.
    New-Segment 8.40 6.48 7.78 6.48 "#1f4e79" $false $false
    New-Segment 7.78 6.48 7.78 11.25 "#1f4e79" $false $false
    New-Segment 7.78 7.35 8.08 7.35 "#1f4e79" $false $true
    New-Segment 7.78 8.55 8.08 8.55 "#1f4e79" $false $true
    New-Segment 7.78 9.95 8.08 9.95 "#1f4e79" $false $true
    New-Segment 7.78 11.25 8.08 11.25 "#1f4e79" $false $true

    # Four loss outputs merge on a short green bus and enter module D.
    New-Segment 19.95 7.35 20.08 7.35 "#2e7d32" $false $false
    New-Segment 19.95 8.55 20.08 8.55 "#2e7d32" $false $false
    New-Segment 19.95 9.95 20.08 9.95 "#2e7d32" $false $false
    New-Segment 19.95 11.25 20.08 11.25 "#2e7d32" $false $false
    New-Segment 3.55 3.15 3.55 3.53 "#94a3b8" $false $false
    New-Segment 9.00 6.00 6.90 6.00 "#94a3b8" $false $false
    New-Segment 7.95 9.00 6.90 9.00 "#94a3b8" $false $false
    New-Segment 22.55 8.85 22.55 9.00 "#94a3b8" $false $false
    New-Segment 12.45 13.88 12.45 14.08 "#94a3b8" $false $false
)

$backpropSegments = @(
    New-Segment 24.43 2.33 24.85 2.33 "#c55a11" $true $false
    New-Segment 24.85 2.33 24.85 12.30 "#c55a11" $true $false
    New-Segment 24.85 12.30 5.00 12.30 "#c55a11" $true $true
)

$visio = $null
$doc = $null
try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    $visio.AlertResponse = 7
    $doc = $visio.Documents.Add("")
    $page = $visio.ActivePage
    $page.Name = Convert-UnicodeEscapes "\u603b\u89c8_Overall"
    $page.PageSheet.CellsU("PageWidth").ResultIU = $pageWidth
    $page.PageSheet.CellsU("PageHeight").ResultIU = $pageHeight

    foreach ($panel in $panels) {
        $centerX = [double]$panel.x
        $centerY = $pageHeight - [double]$panel.y
        $left = $centerX - [double]$panel.w / 2.0
        $right = $centerX + [double]$panel.w / 2.0
        $bottom = $centerY - [double]$panel.h / 2.0
        $top = $centerY + [double]$panel.h / 2.0
        $shape = $page.DrawRectangle($left, $bottom, $right, $top)
        $shape.NameU = [string]$panel.key
        $shape.Text = Convert-UnicodeEscapes ([string]$panel.label)
        Set-CellFormula $shape "FillForegnd" "RGB(250,251,252)"
        Set-CellFormula $shape "FillPattern" "1"
        Set-CellFormula $shape "LineColor" "RGB(190,199,210)"
        Set-CellFormula $shape "LineWeight" "0.65 pt"
        Set-CellFormula $shape "Rounding" "0.025 in"
        Set-CellFormula $shape "Char.Size" "8.5 pt"
        Set-CellFormula $shape "Char.Style" "1"
        Set-CellFormula $shape "Para.HorzAlign" "0"
        Set-CellFormula $shape "VerticalAlign" "0"
        Set-CellFormula $shape "LeftMargin" "0.12 in"
        Set-CellFormula $shape "TopMargin" "0.08 in"
    }

    $shapeMap = @{}
    foreach ($node in $nodes) {
        $centerX = [double]$node.x
        $centerY = $pageHeight - [double]$node.y
        $left = $centerX - [double]$node.w / 2.0
        $right = $centerX + [double]$node.w / 2.0
        $bottom = $centerY - [double]$node.h / 2.0
        $top = $centerY + [double]$node.h / 2.0
        $shape = $page.DrawRectangle($left, $bottom, $right, $top)
        $shape.NameU = [string]$node.key
        $shape.Text = Convert-UnicodeEscapes ([string]$node.label)
        Set-CellFormula $shape "FillForegnd" (Rgb-Formula ([string]$node.fill))
        Set-CellFormula $shape "FillPattern" "1"
        Set-CellFormula $shape "LineColor" "RGB(71,85,105)"
        Set-CellFormula $shape "LineWeight" "0.85 pt"
        Set-CellFormula $shape "Rounding" "0.035 in"
        Set-CellFormula $shape "Char.Size" "7.8 pt"
        Set-CellFormula $shape "Para.HorzAlign" "1"
        Set-CellFormula $shape "VerticalAlign" "1"
        switch ([string]$node.role) {
            "title" {
                Set-CellFormula $shape "FillPattern" "0"
                Set-CellFormula $shape "LinePattern" "0"
                Set-CellFormula $shape "Char.Size" "13 pt"
                Set-CellFormula $shape "Char.Style" "1"
                Set-CellFormula $shape "Para.HorzAlign" "0"
            }
            "subtitle" {
                Set-CellFormula $shape "FillPattern" "0"
                Set-CellFormula $shape "LinePattern" "0"
                Set-CellFormula $shape "Char.Size" "8 pt"
                Set-CellFormula $shape "Para.HorzAlign" "0"
            }
            "legend" {
                Set-CellFormula $shape "FillPattern" "0"
                Set-CellFormula $shape "LinePattern" "0"
                Set-CellFormula $shape "Char.Size" "6.8 pt"
                Set-CellFormula $shape "Para.HorzAlign" "2"
            }
            "annotation" {
                Set-CellFormula $shape "FillPattern" "1"
                Set-CellFormula $shape "LinePattern" "0"
                Set-CellFormula $shape "Char.Size" "6.8 pt"
            }
            "loss" {
                Set-CellFormula $shape "LineColor" "RGB(126,139,156)"
                Set-CellFormula $shape "Char.Size" "7.4 pt"
            }
            "constraint" {
                Set-CellFormula $shape "LineColor" "RGB(100,116,139)"
                Set-CellFormula $shape "Char.Size" "7.2 pt"
            }
            "c_step" {
                Set-CellFormula $shape "LineColor" "RGB(100,116,139)"
                Set-CellFormula $shape "LineWeight" "0.75 pt"
                Set-CellFormula $shape "Char.Size" "6.0 pt"
                Set-CellFormula $shape "Para.HorzAlign" "0"
                Set-CellFormula $shape "LeftMargin" "0.07 in"
                Set-CellFormula $shape "RightMargin" "0.05 in"
            }
            "formula" {
                Set-CellFormula $shape "LineColor" "RGB(100,116,139)"
                Set-CellFormula $shape "LineWeight" "0.75 pt"
                Set-CellFormula $shape "Char.Size" "6.2 pt"
                Set-CellFormula $shape "Para.HorzAlign" "0"
                Set-CellFormula $shape "LeftMargin" "0.09 in"
                Set-CellFormula $shape "RightMargin" "0.06 in"
            }
            "loss_bus" {
                Set-CellFormula $shape "FillForegnd" "RGB(46,125,50)"
                Set-CellFormula $shape "LineColor" "RGB(46,125,50)"
                Set-CellFormula $shape "LineWeight" "0.8 pt"
                Set-CellFormula $shape "Rounding" "0 in"
            }
            "objective" {
                Set-CellFormula $shape "LineColor" "RGB(71,85,105)"
                Set-CellFormula $shape "LineWeight" "1 pt"
                Set-CellFormula $shape "Char.Size" "6.2 pt"
            }
            "objective_detail" {
                Set-CellFormula $shape "LineColor" "RGB(173,183,196)"
                Set-CellFormula $shape "LineWeight" "0.55 pt"
                Set-CellFormula $shape "Rounding" "0.02 in"
                Set-CellFormula $shape "Char.Size" "5.6 pt"
                Set-CellFormula $shape "Para.HorzAlign" "0"
                Set-CellFormula $shape "LeftMargin" "0.08 in"
            }
            "whw" {
                Set-CellFormula $shape "LineColor" "RGB(177,145,70)"
                Set-CellFormula $shape "LineWeight" "0.75 pt"
                Set-CellFormula $shape "Rounding" "0.025 in"
                Set-CellFormula $shape "Char.Size" "7.3 pt"
                Set-CellFormula $shape "Para.HorzAlign" "0"
                Set-CellFormula $shape "LeftMargin" "0.10 in"
                Set-CellFormula $shape "RightMargin" "0.08 in"
            }
            "whw_compact" {
                Set-CellFormula $shape "LineColor" "RGB(177,145,70)"
                Set-CellFormula $shape "LineWeight" "0.75 pt"
                Set-CellFormula $shape "Rounding" "0.025 in"
                Set-CellFormula $shape "Char.Size" "6.5 pt"
                Set-CellFormula $shape "Para.HorzAlign" "0"
                Set-CellFormula $shape "LeftMargin" "0.08 in"
                Set-CellFormula $shape "RightMargin" "0.06 in"
            }
            "whw_wide" {
                Set-CellFormula $shape "LineColor" "RGB(177,145,70)"
                Set-CellFormula $shape "LineWeight" "0.75 pt"
                Set-CellFormula $shape "Rounding" "0.025 in"
                Set-CellFormula $shape "Char.Size" "7.2 pt"
                Set-CellFormula $shape "Para.HorzAlign" "0"
                Set-CellFormula $shape "LeftMargin" "0.10 in"
                Set-CellFormula $shape "RightMargin" "0.08 in"
            }
        }
        $shapeMap[[string]$node.key] = $shape
    }

    foreach ($edge in $edges) {
        $src = $shapeMap[[string]$edge.start]
        $dst = $shapeMap[[string]$edge.end]
        $startPoint = Get-AnchorPoint $src ([string]$edge.from)
        $endPoint = Get-AnchorPoint $dst ([string]$edge.to)
        $line = $page.DrawLine($startPoint[0], $startPoint[1], $endPoint[0], $endPoint[1])
        $line.NameU = "edge_" + [string]$edge.start + "_" + [string]$edge.end
        Set-CellFormula $line "LineColor" (Rgb-Formula ([string]$edge.color))
        Set-CellFormula $line "LineWeight" "0.9 pt"
        if ([bool]$edge.arrow) {
            Set-CellFormula $line "EndArrow" "4"
        } else {
            Set-CellFormula $line "EndArrow" "0"
        }
        if ([bool]$edge.dashed) {
            Set-CellFormula $line "LinePattern" "2"
        }
        if (-not [string]::IsNullOrWhiteSpace([string]$edge.label)) {
            $line.Text = Convert-UnicodeEscapes ([string]$edge.label)
            Set-CellFormula $line "Char.Size" "6 pt"
        }
    }

    $segmentIndex = 0
    foreach ($segment in $constraintSegments) {
        $segmentIndex += 1
        $line = $page.DrawLine(
            [double]$segment.x1,
            ($pageHeight - [double]$segment.y1),
            [double]$segment.x2,
            ($pageHeight - [double]$segment.y2)
        )
        $line.NameU = "constraint_segment_" + $segmentIndex
        Set-CellFormula $line "LineColor" (Rgb-Formula ([string]$segment.color))
        Set-CellFormula $line "LineWeight" "0.75 pt"
        if ([bool]$segment.dashed) {
            Set-CellFormula $line "LinePattern" "2"
        }
        if ([bool]$segment.arrow) {
            Set-CellFormula $line "EndArrow" "4"
        } else {
            Set-CellFormula $line "EndArrow" "0"
        }
    }

    $backpropIndex = 0
    foreach ($segment in $backpropSegments) {
        $backpropIndex += 1
        $line = $page.DrawLine(
            [double]$segment.x1,
            ($pageHeight - [double]$segment.y1),
            [double]$segment.x2,
            ($pageHeight - [double]$segment.y2)
        )
        $line.NameU = "backprop_segment_" + $backpropIndex
        Set-CellFormula $line "LineColor" (Rgb-Formula ([string]$segment.color))
        Set-CellFormula $line "LineWeight" "0.9 pt"
        if ([bool]$segment.dashed) {
            Set-CellFormula $line "LinePattern" "2"
        }
        if ([bool]$segment.arrow) {
            Set-CellFormula $line "EndArrow" "4"
        } else {
            Set-CellFormula $line "EndArrow" "0"
        }
    }

    # Page 2: step-by-step explanation of module B.
    $bPageWidth = 21.0
    $bPageHeight = 10.5
    $pageB = $doc.Pages.Add()
    $pageB.Name = Convert-UnicodeEscapes "\u6a21\u5757B_\u9010\u6b65\u8bf4\u660e"
    $pageB.PageSheet.CellsU("PageWidth").ResultIU = $bPageWidth
    $pageB.PageSheet.CellsU("PageHeight").ResultIU = $bPageHeight

    $bPanels = @(
        New-Panel "b_forward_panel" 10.50 2.75 20.20 3.35 "B \u6a21\u5757\u7279\u5f81\u6d41\uff1a\u53cc\u5411\u57df\u7ffb\u8bd1 \u2192 \u53cc\u5224\u522b\u5668 \u2192 \u5bf9\u6297\u4f18\u5316"
        New-Panel "b_explain_panel" 10.50 7.25 20.20 3.75 "B \u6a21\u5757\u9010\u6b65 WHAT / HOW / WHY"
    )
    $bNodes = @(
        New-Node "b_title" 8.10 0.32 15.40 0.40 "\u6a21\u5757 B \u8be6\u89e3\uff1a\u53cc\u5411\u57df\u7ffb\u8bd1\u4e0e\u53cc\u5224\u522b\u5668" "#ffffff" "title"
        New-Node "b_subtitle" 8.10 0.75 15.40 0.28 "\u84dd\u8272\u5b9e\u7ebf\uff1a\u7279\u5f81\u524d\u5411\u6d41    \u7eff\u8272\u5b9e\u7ebf\uff1a\u635f\u5931/\u8f93\u51fa\u6c47\u603b    \u6a59\u8272\u865a\u7ebf\uff1a\u53cd\u5411\u4f20\u64ad" "#ffffff" "subtitle"

        New-Node "b_ft" 1.55 2.10 1.50 0.64 "\u76ee\u6807\u57df\u7279\u5f81\nF_t" "#dcecff"
        New-Node "b_gt2s" 3.85 2.10 1.85 0.72 "G_t2s\n\u76ee\u6807 \u2192 \u6e90\u57df" "#fff0d9"
        New-Node "b_fts" 6.25 2.10 1.85 0.72 "source-like\nF_ts = G_t2s(F_t)" "#fff8ea"
        New-Node "b_ds" 8.75 2.10 2.00 0.76 "\u4e3b\u5224\u522b\u5668 D_s\n\u771f F_s / \u5047 F_ts" "#fde8e7"
        New-Node "b_advs" 11.25 2.10 1.75 0.70 "L_adv^s\nCE(D_s(F_ts),1)" "#eef1f5" "loss"

        New-Node "b_fs" 1.55 3.38 1.50 0.64 "\u6e90\u57df\u7279\u5f81\nF_s" "#dcecff"
        New-Node "b_gs2t" 3.85 3.38 1.85 0.72 "G_s2t\n\u6e90\u57df \u2192 \u76ee\u6807" "#fff0d9"
        New-Node "b_fst" 6.25 3.38 1.85 0.72 "target-like\nF_st = G_s2t(F_s)" "#fff8ea"
        New-Node "b_dt" 8.75 3.38 2.00 0.76 "\u8f85\u52a9\u5224\u522b\u5668 D_t\n\u771f F_t / \u5047 F_st" "#fde8e7"
        New-Node "b_advt" 11.25 3.38 1.75 0.70 "L_adv^t\nCE(D_t(F_st),1)" "#eef1f5" "loss"

        New-Node "b_output" 14.15 2.74 2.65 1.30 "B \u8f93\u51fa\nF_ts\uff0cF_st\nL_adv^s\uff0cL_adv^t" "#e5f5ec"
        New-Node "b_to_c" 17.55 2.05 2.85 0.86 "\u9001\u5165\u6a21\u5757 C\nF_s,F_t,F_ts,F_st,y" "#eee9fb"
        New-Node "b_joint" 17.55 3.45 2.85 0.86 "\u9001\u5165\u6a21\u5757 D\nL_adv^s,L_adv^t" "#e8f3e8"
        New-Node "b_bp_label" 10.50 4.70 9.50 0.26 "\u53cd\u5411\u4f20\u64ad\uff1a\u5bf9\u6297\u635f\u5931\u66f4\u65b0 G_t2s/G_s2t\uff0c\u5224\u522b\u635f\u5931\u4ea4\u66ff\u66f4\u65b0 D_s/D_t" "#ffffff" "annotation"

        New-Node "b_whw1" 2.25 7.30 3.75 2.65 "B1 \u65b9\u5411\u4e0e\u8f93\u5165\u5b9a\u4e49\nWHAT\uff5c\u5b9a\u4e49 F_t\u2192\u6e90\u57df\u4e3b\u8def\u548c F_s\u2192\u76ee\u6807\u57df\u8f85\u52a9\u8def\u5f84\u3002\nHOW \uff5c\u4f7f\u7528 A \u6a21\u5757\u5df2\u5bf9\u9f50\u7684 F_s/F_t\uff0c\u56fa\u5b9a\u4e24\u4e2a\u6620\u5c04\u65b9\u5411\u3002\nWHY \uff5c\u63a8\u7406\u9700\u5c06\u6076\u52a3\u5929\u6c14\u7279\u5f81\u8f6c\u4e3a\u66f4\u7a33\u5b9a\u7684 source-like \u8868\u793a\uff1b\u53cd\u5411\u8def\u5f84\u7528\u4e8e\u7ea6\u675f\u53ef\u9006\u6027\u3002" "#fff9e8" "whw_detail"
        New-Node "b_whw2" 6.35 7.30 3.75 2.65 "B2 \u53cc\u5411\u6b8b\u5dee\u7ffb\u8bd1\u5668\nWHAT\uff5c\u4ec5\u6539\u53d8\u57df\u98ce\u683c\uff0c\u5c3d\u91cf\u4fdd\u7559\u76ee\u6807\u8bed\u4e49\u3002\nHOW \uff5cG_t2s/G_s2t \u91c7\u7528\u6b8b\u5dee MLP\u3001LayerNorm\u3001Dropout \u548c\u53d7\u63a7 residual scale\u3002\nWHY \uff5c\u7279\u5f81\u7ea7\u6620\u5c04\u6bd4\u56fe\u50cf\u751f\u6210\u66f4\u8f7b\u91cf\uff1b\u6b8b\u5dee\u8bbe\u8ba1\u53ef\u907f\u514d\u5bf9\u8239\u4f53\u5f31\u7279\u5f81\u5927\u5e45\u6539\u5199\u3002" "#fff9e8" "whw_detail"
        New-Node "b_whw3" 10.45 7.30 3.75 2.65 "B3 \u751f\u6210\u7279\u5f81 F_ts/F_st\nWHAT\uff5c\u5f97\u5230\u4e24\u4e2a\u5e26\u65b9\u5411\u7684\u57df\u7ffb\u8bd1\u7ed3\u679c\u3002\nHOW \uff5cF_ts=G_t2s(F_t)\uff1bF_st=G_s2t(F_s)\u3002\nWHY \uff5cF_ts \u662f\u6076\u52a3\u5929\u6c14\u63a8\u7406\u4e3b\u8def\u4f7f\u7528\u7684\u6e90\u57df\u5316\u7279\u5f81\uff1bF_st \u4e3a Cycle\u3001Identity \u548c\u7c7b\u522b\u53cd\u9988\u63d0\u4f9b\u53cd\u5411\u8bc1\u636e\u3002" "#fff9e8" "whw_detail"
        New-Node "b_whw4" 14.55 7.30 3.75 2.65 "B4 \u53cc\u5224\u522b\u5668 D_s/D_t\nWHAT\uff5c\u5224\u65ad\u751f\u6210\u7279\u5f81\u662f\u5426\u7b26\u5408\u5bf9\u5e94\u771f\u5b9e\u57df\u5206\u5e03\u3002\nHOW \uff5cD_s \u533a\u5206 F_s/F_ts\uff1bD_t \u533a\u5206 F_t/F_st\uff1b\u4f7f\u7528\u771f/\u5047\u4ea4\u53c9\u71b5\u3002\nWHY \uff5c\u53cc\u65b9\u5411\u5224\u522b\u53ef\u9632\u6b62\u5355\u5411\u5bf9\u9f50\u584c\u7f29\uff0c\u63d0\u9ad8\u9ed1\u5929\u3001\u96e8\u96fe\u4e0e\u6d77\u6742\u6ce2\u4e0b\u7684\u5206\u5e03\u7a33\u5b9a\u6027\u3002" "#fff9e8" "whw_detail"
        New-Node "b_whw5" 18.65 7.30 3.75 2.65 "B5 \u5bf9\u6297\u4f18\u5316\u4e0e\u8f93\u51fa\nWHAT\uff5c\u8ba9\u751f\u6210\u5668\u6b3a\u9a97\u5224\u522b\u5668\uff0c\u5e76\u5c06\u7ffb\u8bd1\u7279\u5f81\u9001\u5165 C\u3002\nHOW \uff5cL_adv=CE(D(F_fake),1)\uff1b\u4f7f\u7528\u5bf9\u6297\u9884\u70ed/\u6e10\u589e\u4e0e\u4ea4\u66ff\u66f4\u65b0\u3002\nWHY \uff5c\u907f\u514d\u8bad\u7ec3\u521d\u671f\u5224\u522b\u5668\u8fc7\u5f3a\u5f15\u8d77\u635f\u5931\u9707\u8361\uff0c\u540c\u65f6\u4e3a C \u6a21\u5757\u63d0\u4f9b\u660e\u786e\u7684 F_s/F_t/F_ts/F_st \u8f93\u5165\u3002" "#fff9e8" "whw_detail"
    )
    $bEdges = @(
        New-Edge "b_ft" "b_gt2s" "" $false "E" "W" "#1f4e79"
        New-Edge "b_gt2s" "b_fts" "" $false "E" "W" "#1f4e79"
        New-Edge "b_fts" "b_ds" "" $false "E" "W" "#1f4e79"
        New-Edge "b_ds" "b_advs" "" $false "E" "W" "#2e7d32"
        New-Edge "b_fs" "b_gs2t" "" $false "E" "W" "#1f4e79"
        New-Edge "b_gs2t" "b_fst" "" $false "E" "W" "#1f4e79"
        New-Edge "b_fst" "b_dt" "" $false "E" "W" "#1f4e79"
        New-Edge "b_dt" "b_advt" "" $false "E" "W" "#2e7d32"
        New-Edge "b_advs" "b_output" "" $false "E" "W" "#2e7d32"
        New-Edge "b_advt" "b_output" "" $false "E" "W" "#2e7d32"
        New-Edge "b_output" "b_to_c" "" $false "E" "W" "#1f4e79"
        New-Edge "b_output" "b_joint" "" $false "E" "W" "#2e7d32"
    )
    $bSegments = @(
        New-Segment 18.98 3.45 20.55 3.45 "#c55a11" $true $false
        New-Segment 20.55 3.25 20.55 4.82 "#c55a11" $true $false
        New-Segment 20.55 4.82 1.00 4.82 "#c55a11" $true $true
    )
    foreach ($panelItem in $bPanels) { Add-DetailPanel $pageB $bPageHeight $panelItem | Out-Null }
    $bShapeMap = @{}
    foreach ($nodeItem in $bNodes) { $bShapeMap[[string]$nodeItem.key] = Add-DetailNode $pageB $bPageHeight $nodeItem }
    foreach ($edgeItem in $bEdges) { Add-DetailEdge $pageB $bShapeMap $edgeItem | Out-Null }
    $detailSegmentIndex = 0
    foreach ($segmentItem in $bSegments) { $detailSegmentIndex += 1; Add-DetailSegment $pageB $bPageHeight $segmentItem $detailSegmentIndex | Out-Null }

    # The former branch-heavy module-C page is retained only as an optional
    # compatibility page. The default output integrates C1-C4 into Overall.
    $pageC = $null
    if ($IncludeLegacyModuleCPage) {
        $cPageWidth = 24.0
        $cPageHeight = 14.0
        $pageC = $doc.Pages.Add()
        $pageC.Name = Convert-UnicodeEscapes "\u6a21\u5757C_\u65e7\u7248\u5206\u652f\u8be6\u89e3"
        $pageC.PageSheet.CellsU("PageWidth").ResultIU = $cPageWidth
        $pageC.PageSheet.CellsU("PageHeight").ResultIU = $cPageHeight

    $cPanels = @(
        New-Panel "c_id_panel" 7.40 2.65 14.20 2.05 "C1  Identity \u8eab\u4efd\u4fdd\u6301\u5206\u652f"
        New-Panel "c_cycle_panel" 7.40 5.05 14.20 2.35 "C2  Cycle \u53cc\u5411\u95ed\u73af\u91cd\u5efa\u5206\u652f"
        New-Panel "c_contrast_panel" 7.40 7.95 14.20 2.80 "C3  \u914d\u5bf9\u5bf9\u6bd4 + \u7c7b\u522b\u539f\u578b\u5bf9\u6bd4\u5206\u652f"
        New-Panel "c_gcls_panel" 7.40 10.55 14.20 1.80 "C4  \u751f\u6210\u7279\u5f81\u5206\u7c7b\u53cd\u9988\u5206\u652f"
    )
    $cNodes = @(
        New-Node "c_title" 8.50 0.30 16.20 0.40 "\u6a21\u5757 C \u8be6\u89e3\uff1a\u8bed\u4e49\u4fdd\u6301\u4e0e\u7c7b\u522b\u611f\u77e5\u53cd\u9988" "#ffffff" "title"
        New-Node "c_subtitle" 8.50 0.72 16.20 0.28 "\u56db\u4e2a\u5206\u652f\u5e76\u884c\u8bfb\u53d6 B \u8f93\u51fa\uff1aF_s\u3001F_t\u3001F_ts\u3001F_st\uff0c\u5e76\u4f7f\u7528 y_s/y_t\u3001C\u3001G_t2s/G_s2t" "#ffffff" "subtitle"
        New-Node "c_input_summary" 7.40 1.18 13.20 0.48 "C \u5171\u4eab\u8f93\u5165\uff1a{F_s,F_t,F_ts,F_st,y_s,y_t,C,G_t2s,G_s2t}" "#e7eef7" "lane"

        New-Node "c_id_input" 1.35 2.65 1.55 0.64 "F_s / F_t" "#dcecff"
        New-Node "c_id_map" 4.15 2.65 3.00 0.82 "G_t2s(F_s)\nG_s2t(F_t)" "#fff0d9"
        New-Node "c_id_feat" 7.15 2.65 2.10 0.72 "F_s^id / F_t^id" "#fff8ea"
        New-Node "c_id_cmp" 10.05 2.65 2.40 0.78 "L1 \u5dee\u5f02\nvs. F_s / F_t" "#e9f5e6"
        New-Node "c_id_loss" 12.85 2.65 1.70 0.66 "L_id" "#eef1f5" "loss"
        New-Node "c_id_whw" 19.25 2.65 8.60 1.95 "C1 Identity\nWHAT\uff5c\u8f93\u5165\u5df2\u5c5e\u4e8e\u76ee\u6807\u98ce\u683c\u65f6\uff0c\u751f\u6210\u5668\u5e94\u8fd1\u4f3c\u6052\u7b49\u6620\u5c04\u3002\nHOW \uff5c\u8ba1\u7b97 G_t2s(F_s)\u4e0e F_s\u3001G_s2t(F_t)\u4e0e F_t \u7684 L1 \u8ddd\u79bb\u3002\nWHY \uff5c\u6d77\u4e0a\u76ee\u6807\u7684\u70ed\u7ed3\u6784\u548c\u5f31\u8f6e\u5ed3\u5df2\u5f88\u73cd\u8d35\uff0c\u4e0d\u5e94\u88ab\u57df\u7ffb\u8bd1\u65e0\u6545\u6539\u5199\u3002" "#fff9e8" "whw_detail"

        New-Node "c_cyc_fts" 1.15 4.68 1.30 0.54 "F_ts" "#dcecff"
        New-Node "c_cyc_gs2t" 3.35 4.68 1.70 0.62 "G_s2t" "#fff0d9"
        New-Node "c_cyc_fthat" 5.55 4.68 1.65 0.62 "F_t_hat" "#fff8ea"
        New-Node "c_cyc_cmp_t" 8.05 4.68 2.10 0.66 "||F_t_hat-F_t||_1" "#e9f5e6"
        New-Node "c_cyc_fst" 1.15 5.55 1.30 0.54 "F_st" "#dcecff"
        New-Node "c_cyc_gt2s" 3.35 5.55 1.70 0.62 "G_t2s" "#fff0d9"
        New-Node "c_cyc_fshat" 5.55 5.55 1.65 0.62 "F_s_hat" "#fff8ea"
        New-Node "c_cyc_cmp_s" 8.05 5.55 2.10 0.66 "||F_s_hat-F_s||_1" "#e9f5e6"
        New-Node "c_cyc_loss" 11.55 5.12 2.00 0.72 "L_cyc\n\u4e24\u8def L1 \u4e4b\u548c" "#eef1f5" "loss"
        New-Node "c_cyc_whw" 19.25 5.05 8.60 2.20 "C2 Cycle\nWHAT\uff5c\u7ea6\u675f\u7279\u5f81\u7ecf\u8fc7\u53cc\u5411\u7ffb\u8bd1\u540e\u80fd\u56de\u5230\u539f\u57df\u3002\nHOW \uff5cF_ts\u2192G_s2t\u2192F_t_hat\u2248F_t\uff1bF_st\u2192G_t2s\u2192F_s_hat\u2248F_s\uff0c\u4f7f\u7528 L1 \u91cd\u5efa\u635f\u5931\u3002\nWHY \uff5c\u7eaf\u5bf9\u6297\u5b66\u4e60\u53ef\u80fd\u53ea\u6539\u5206\u5e03\u800c\u4e22\u5931\u8239\u53ea\u5c3a\u5bf8\u3001\u8f6e\u5ed3\u548c\u5c40\u90e8\u7ed3\u6784\uff1b\u95ed\u73af\u7ea6\u675f\u963b\u6b62\u4fe1\u606f\u4e0d\u53ef\u9006\u4e22\u5931\u3002" "#fff9e8" "whw_detail"

        New-Node "c_proto_real" 1.65 7.48 2.25 0.62 "F_s+y_s / F_t+y_t" "#dcecff"
        New-Node "c_proto_build" 4.45 7.48 2.15 0.70 "\u6309\u7c7b\u6c42\u5747\u503c\nP_s^c / P_t^c" "#eee9fb"
        New-Node "c_proto_anchor" 7.25 7.48 2.45 0.70 "F_ts+P_s^c+y_t\nF_st+P_t^c+y_s" "#fff8ea"
        New-Node "c_proto_ce" 10.15 7.48 2.10 0.70 "CE(sim/\u03c4)" "#e9f5e6"
        New-Node "c_proto_loss" 12.85 7.48 1.70 0.64 "L_proto" "#eef1f5" "loss"
        New-Node "c_pair_input" 2.15 8.52 3.20 0.62 "(F_ts,F_s) / (F_st,F_t)" "#dcecff"
        New-Node "c_pair_pce" 6.00 8.52 2.20 0.66 "PCE \u914d\u5bf9\u5bf9\u6bd4" "#e9f5e6"
        New-Node "c_pair_loss" 10.10 8.52 1.70 0.64 "L_con" "#eef1f5" "loss"
        New-Node "c_con_whw" 19.25 7.95 8.60 2.60 "C3 \u914d\u5bf9/\u539f\u578b\u5bf9\u6bd4\nWHAT\uff5c\u4fdd\u7559\u6837\u672c\u5bf9\u5e94\u5173\u7cfb\u548c\u7c7b\u522b\u7c07\u7ed3\u6784\u3002\nHOW \uff5cPCE \u5bf9\u9f50\u7ffb\u8bd1/\u771f\u5b9e\u7279\u5f81\uff1b\u518d\u7531 F_s/F_t \u6309\u7c7b\u6c42 P_s^c/P_t^c\uff0c\u5bf9 F_ts/F_st \u6267\u884c\u6e29\u5ea6\u7f29\u653e\u539f\u578b\u4ea4\u53c9\u71b5\u3002\nWHY \uff5c\u4e0d\u540c\u8239\u578b\u548c\u6d77\u4e0a\u5c0f\u76ee\u6807\u7279\u5f81\u63a5\u8fd1\uff0c\u5355\u7eaf\u57df\u5bf9\u9f50\u6613\u628a\u5f02\u7c7b\u62c9\u5230\u4e00\u8d77\uff1b\u539f\u578b\u5bf9\u6bd4\u660e\u786e\u62c9\u8fd1\u540c\u7c7b\u3001\u63a8\u8fdc\u5f02\u7c7b\u3002" "#fff9e8" "whw_detail"

        New-Node "c_gcls_input" 1.55 10.55 2.10 0.64 "F_ts / F_st" "#dcecff"
        New-Node "c_gcls_c" 4.15 10.55 1.55 0.64 "\u5206\u7c7b\u5668 C" "#ece9fb"
        New-Node "c_gcls_logits" 6.75 10.55 1.85 0.64 "logits_ts / logits_st" "#fff8ea"
        New-Node "c_gcls_ce" 9.45 10.55 2.15 0.70 "CE(logits_ts,y_t)\nCE(logits_st,y_s)" "#e9f5e6"
        New-Node "c_gcls_loss" 12.55 10.55 1.75 0.64 "L_gcls" "#eef1f5" "loss"
        New-Node "c_gcls_whw" 19.25 10.55 8.60 1.70 "C4 \u751f\u6210\u7279\u5f81\u5206\u7c7b\u53cd\u9988\nWHAT\uff5c\u68c0\u67e5\u7ffb\u8bd1\u540e\u7279\u5f81\u662f\u5426\u4ecd\u7136\u53ef\u5206\u7c7b\u3002\nHOW \uff5cC(F_ts) \u4f7f\u7528 y_t\uff0cC(F_st) \u4f7f\u7528 y_s\uff0c\u5bf9\u4e24\u8def logits \u8ba1\u7b97\u4ea4\u53c9\u71b5\u3002\nWHY \uff5c\u5bf9\u6297\u57df\u5bf9\u9f50\u53ea\u4fdd\u8bc1\u201c\u50cf\u54ea\u4e2a\u57df\u201d\uff0c\u4e0d\u4fdd\u8bc1\u201c\u8fd8\u662f\u54ea\u4e00\u7c7b\u201d\uff1b\u5206\u7c7b\u53cd\u9988\u76f4\u63a5\u4fdd\u62a4\u4efb\u52a1\u8bed\u4e49\u3002" "#fff9e8" "whw_detail"

        New-Node "c_output_all" 9.80 12.20 8.40 0.72 "C \u8f93\u51fa\uff1a{L_id, L_cyc, L_con, L_proto, L_gcls}" "#e5f5ec"
        New-Node "c_to_d" 18.80 12.20 7.00 0.72 "\u7eff\u8272\u6c47\u603b\uff1a\u52a0\u5165\u6a21\u5757 D \u7684 L_total" "#e8f3e8"
        New-Node "c_bp_label" 12.00 13.15 10.80 0.26 "\u6a59\u8272\u865a\u7ebf\uff1a\u68af\u5ea6\u56de\u4f20\u66f4\u65b0 C\u3001G_t2s/G_s2t\u3001TAL \u4e0e\u7f16\u7801\u5668" "#ffffff" "annotation"
    )
    $cEdges = @(
        New-Edge "c_id_input" "c_id_map" "" $false "E" "W" "#1f4e79"
        New-Edge "c_id_map" "c_id_feat" "" $false "E" "W" "#1f4e79"
        New-Edge "c_id_feat" "c_id_cmp" "" $false "E" "W" "#1f4e79"
        New-Edge "c_id_cmp" "c_id_loss" "" $false "E" "W" "#2e7d32"
        New-Edge "c_cyc_fts" "c_cyc_gs2t" "" $false "E" "W" "#1f4e79"
        New-Edge "c_cyc_gs2t" "c_cyc_fthat" "" $false "E" "W" "#1f4e79"
        New-Edge "c_cyc_fthat" "c_cyc_cmp_t" "" $false "E" "W" "#1f4e79"
        New-Edge "c_cyc_fst" "c_cyc_gt2s" "" $false "E" "W" "#1f4e79"
        New-Edge "c_cyc_gt2s" "c_cyc_fshat" "" $false "E" "W" "#1f4e79"
        New-Edge "c_cyc_fshat" "c_cyc_cmp_s" "" $false "E" "W" "#1f4e79"
        New-Edge "c_cyc_cmp_t" "c_cyc_loss" "" $false "E" "W" "#2e7d32"
        New-Edge "c_cyc_cmp_s" "c_cyc_loss" "" $false "E" "W" "#2e7d32"
        New-Edge "c_proto_real" "c_proto_build" "" $false "E" "W" "#1f4e79"
        New-Edge "c_proto_build" "c_proto_anchor" "" $false "E" "W" "#1f4e79"
        New-Edge "c_proto_anchor" "c_proto_ce" "" $false "E" "W" "#1f4e79"
        New-Edge "c_proto_ce" "c_proto_loss" "" $false "E" "W" "#2e7d32"
        New-Edge "c_pair_input" "c_pair_pce" "" $false "E" "W" "#1f4e79"
        New-Edge "c_pair_pce" "c_pair_loss" "" $false "E" "W" "#2e7d32"
        New-Edge "c_gcls_input" "c_gcls_c" "" $false "E" "W" "#1f4e79"
        New-Edge "c_gcls_c" "c_gcls_logits" "" $false "E" "W" "#1f4e79"
        New-Edge "c_gcls_logits" "c_gcls_ce" "" $false "E" "W" "#1f4e79"
        New-Edge "c_gcls_ce" "c_gcls_loss" "" $false "E" "W" "#2e7d32"
        New-Edge "c_output_all" "c_to_d" "" $false "E" "W" "#2e7d32"
    )
    $cSegments = @(
        New-Segment 14.40 2.65 14.40 12.20 "#2e7d32" $false $false
        New-Segment 13.70 2.65 14.40 2.65 "#2e7d32" $false $false
        New-Segment 12.55 5.12 14.40 5.12 "#2e7d32" $false $false
        New-Segment 13.70 7.48 14.40 7.48 "#2e7d32" $false $false
        New-Segment 10.95 8.52 14.40 8.52 "#2e7d32" $false $false
        New-Segment 13.43 10.55 14.40 10.55 "#2e7d32" $false $false
        New-Segment 14.40 12.20 14.00 12.20 "#2e7d32" $false $true
        New-Segment 23.50 12.55 23.50 13.38 "#c55a11" $true $false
        New-Segment 23.50 13.38 0.80 13.38 "#c55a11" $true $true
    )
        foreach ($panelItem in $cPanels) { Add-DetailPanel $pageC $cPageHeight $panelItem | Out-Null }
        $cShapeMap = @{}
        foreach ($nodeItem in $cNodes) { $cShapeMap[[string]$nodeItem.key] = Add-DetailNode $pageC $cPageHeight $nodeItem }
        foreach ($edgeItem in $cEdges) { Add-DetailEdge $pageC $cShapeMap $edgeItem | Out-Null }
        $detailSegmentIndex = 100
        foreach ($segmentItem in $cSegments) { $detailSegmentIndex += 1; Add-DetailSegment $pageC $cPageHeight $segmentItem $detailSegmentIndex | Out-Null }
    }

    if (Test-Path -LiteralPath $OutputPath) {
        Remove-Item -LiteralPath $OutputPath -Force
    }
    $doc.SaveAs($OutputPath)
    Write-Output "Wrote $OutputPath"
    if (-not [string]::IsNullOrWhiteSpace($PreviewPath)) {
        if (Test-Path -LiteralPath $PreviewPath) {
            Remove-Item -LiteralPath $PreviewPath -Force
        }
        $page.Export($PreviewPath)
        $previewBase = [System.IO.Path]::Combine(
            [System.IO.Path]::GetDirectoryName($PreviewPath),
            [System.IO.Path]::GetFileNameWithoutExtension($PreviewPath)
        )
        $previewExtension = [System.IO.Path]::GetExtension($PreviewPath)
        $pageB.Export($previewBase + "_module_B" + $previewExtension)
        if ($pageC -ne $null) {
            $pageC.Export($previewBase + "_module_C_legacy" + $previewExtension)
        }
        Write-Output "Exported $PreviewPath"
    }
} finally {
    if ($doc -ne $null) {
        $doc.Close()
    }
    if ($visio -ne $null) {
        $visio.Quit()
    }
}
