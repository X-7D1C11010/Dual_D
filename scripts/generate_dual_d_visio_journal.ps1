param(
    [string]$OutputPath = "",
    [string]$PreviewPath = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path (Resolve-Path ".").Path "docs\dual_d_framework_flow_journal.vsdx"
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$outputDir = [System.IO.Path]::GetDirectoryName($OutputPath)
if (-not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

if (-not [string]::IsNullOrWhiteSpace($PreviewPath)) {
    $PreviewPath = [System.IO.Path]::GetFullPath($PreviewPath)
    $previewDir = [System.IO.Path]::GetDirectoryName($PreviewPath)
    if (-not (Test-Path -LiteralPath $previewDir)) {
        New-Item -ItemType Directory -Path $previewDir | Out-Null
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
        # Keep generation compatible with different Visio editions.
    }
}

function New-ShapeDef {
    param(
        [string]$Key,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [string]$Text,
        [string]$Type = "process",
        [string]$Fill = "#ffffff"
    )
    return @{
        key = $Key; x = $X; y = $Y; w = $W; h = $H
        text = $Text; type = $Type; fill = $Fill
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

function Add-JournalShape {
    param($Page, [double]$PageHeight, $Def)

    $x = [double]$Def.x
    $y = $PageHeight - [double]$Def.y
    $w = [double]$Def.w
    $h = [double]$Def.h
    $type = [string]$Def.type

    if ($type -eq "decision") {
        [double[]]$points = @(
            $x, ($y + $h / 2.0),
            ($x + $w / 2.0), $y,
            $x, ($y - $h / 2.0),
            ($x - $w / 2.0), $y,
            $x, ($y + $h / 2.0)
        )
        $shape = $Page.DrawPolyline([ref]$points, 0)
    } else {
        $shape = $Page.DrawRectangle(
            ($x - $w / 2.0),
            ($y - $h / 2.0),
            ($x + $w / 2.0),
            ($y + $h / 2.0)
        )
    }

    $shape.NameU = [string]$Def.key
    $shape.Text = Convert-UnicodeEscapes ([string]$Def.text)
    Set-CellFormula $shape "Char.Font" 'FONT("Microsoft YaHei")'
    Set-CellFormula $shape "Char.Color" "RGB(31,41,55)"
    Set-CellFormula $shape "FillForegnd" (Rgb-Formula ([string]$Def.fill))
    Set-CellFormula $shape "FillPattern" "1"
    Set-CellFormula $shape "LineColor" "RGB(71,85,105)"
    Set-CellFormula $shape "LineWeight" "0.75 pt"
    Set-CellFormula $shape "Para.HorzAlign" "1"
    Set-CellFormula $shape "VerticalAlign" "1"
    Set-CellFormula $shape "LeftMargin" "0.06 in"
    Set-CellFormula $shape "RightMargin" "0.06 in"
    Set-CellFormula $shape "TopMargin" "0.04 in"
    Set-CellFormula $shape "BottomMargin" "0.04 in"
    Set-CellFormula $shape "Char.Size" "8.0 pt"
    Set-CellFormula $shape "Rounding" "0.035 in"

    switch ($type) {
        "panel" {
            Set-CellFormula $shape "FillForegnd" "RGB(250,251,252)"
            Set-CellFormula $shape "LineColor" "RGB(174,185,199)"
            Set-CellFormula $shape "LineWeight" "0.65 pt"
            Set-CellFormula $shape "Char.Size" "9.5 pt"
            Set-CellFormula $shape "Char.Style" "1"
            Set-CellFormula $shape "Para.HorzAlign" "0"
            Set-CellFormula $shape "VerticalAlign" "0"
            Set-CellFormula $shape "LeftMargin" "0.10 in"
            Set-CellFormula $shape "TopMargin" "0.06 in"
            Set-CellFormula $shape "Rounding" "0.02 in"
        }
        "lane" {
            Set-CellFormula $shape "FillForegnd" (Rgb-Formula ([string]$Def.fill))
            Set-CellFormula $shape "LineColor" "RGB(210,217,226)"
            Set-CellFormula $shape "LineWeight" "0.4 pt"
            Set-CellFormula $shape "Char.Size" "8.3 pt"
            Set-CellFormula $shape "Char.Style" "1"
            Set-CellFormula $shape "Para.HorzAlign" "0"
            Set-CellFormula $shape "VerticalAlign" "0"
            Set-CellFormula $shape "LeftMargin" "0.08 in"
            Set-CellFormula $shape "TopMargin" "0.04 in"
            Set-CellFormula $shape "Rounding" "0 in"
        }
        "title" {
            Set-CellFormula $shape "FillPattern" "0"
            Set-CellFormula $shape "LinePattern" "0"
            Set-CellFormula $shape "Char.Size" "14 pt"
            Set-CellFormula $shape "Char.Style" "1"
            Set-CellFormula $shape "Para.HorzAlign" "0"
        }
        "legend" {
            Set-CellFormula $shape "FillPattern" "0"
            Set-CellFormula $shape "LinePattern" "0"
            Set-CellFormula $shape "Char.Size" "7.2 pt"
            Set-CellFormula $shape "Para.HorzAlign" "2"
        }
        "main" {
            Set-CellFormula $shape "Char.Size" "7.5 pt"
            Set-CellFormula $shape "LineWeight" "0.85 pt"
        }
        "terminal" {
            Set-CellFormula $shape "Char.Size" "7.4 pt"
            Set-CellFormula $shape "Rounding" "0.22 in"
            Set-CellFormula $shape "LineWeight" "0.85 pt"
        }
        "process" {
            Set-CellFormula $shape "Char.Size" "7.2 pt"
            Set-CellFormula $shape "Para.HorzAlign" "0"
            Set-CellFormula $shape "LeftMargin" "0.07 in"
        }
        "effect" {
            Set-CellFormula $shape "Char.Size" "7.0 pt"
            Set-CellFormula $shape "LineColor" "RGB(109,76,140)"
            Set-CellFormula $shape "Char.Style" "2"
        }
        "decision" {
            Set-CellFormula $shape "FillForegnd" "RGB(255,246,219)"
            Set-CellFormula $shape "LineColor" "RGB(140,109,31)"
            Set-CellFormula $shape "LineWeight" "0.9 pt"
            Set-CellFormula $shape "Char.Size" "6.7 pt"
        }
        "loss" {
            Set-CellFormula $shape "FillForegnd" "RGB(232,243,232)"
            Set-CellFormula $shape "LineColor" "RGB(46,125,50)"
            Set-CellFormula $shape "LineWeight" "0.9 pt"
            Set-CellFormula $shape "Char.Size" "7.0 pt"
            Set-CellFormula $shape "Char.Style" "1"
            Set-CellFormula $shape "Rounding" "0.12 in"
        }
        "aggregate" {
            Set-CellFormula $shape "FillForegnd" "RGB(221,238,224)"
            Set-CellFormula $shape "LineColor" "RGB(46,125,50)"
            Set-CellFormula $shape "LineWeight" "1.0 pt"
            Set-CellFormula $shape "Char.Size" "7.2 pt"
            Set-CellFormula $shape "Char.Style" "1"
        }
        "note" {
            Set-CellFormula $shape "FillPattern" "0"
            Set-CellFormula $shape "LinePattern" "0"
            Set-CellFormula $shape "Char.Size" "6.5 pt"
            Set-CellFormula $shape "Para.HorzAlign" "0"
        }
        "formula" {
            Set-CellFormula $shape "Char.Size" "8.3 pt"
            Set-CellFormula $shape "Para.HorzAlign" "0"
            Set-CellFormula $shape "LeftMargin" "0.10 in"
            Set-CellFormula $shape "RightMargin" "0.08 in"
            Set-CellFormula $shape "TopMargin" "0.06 in"
            Set-CellFormula $shape "BottomMargin" "0.06 in"
        }
    }
    return $shape
}

function Add-Arrow {
    param(
        $Page,
        $ShapeMap,
        [string]$FromKey,
        [string]$ToKey,
        [string]$FromAnchor = "E",
        [string]$ToAnchor = "W",
        [string]$Color = "#1f4e79",
        [bool]$Dashed = $false,
        [string]$Label = ""
    )
    $from = Get-AnchorPoint $ShapeMap[$FromKey] $FromAnchor
    $to = Get-AnchorPoint $ShapeMap[$ToKey] $ToAnchor
    $line = $Page.DrawLine($from[0], $from[1], $to[0], $to[1])
    Set-CellFormula $line "LineColor" (Rgb-Formula $Color)
    Set-CellFormula $line "LineWeight" "0.95 pt"
    Set-CellFormula $line "EndArrow" "4"
    if ($Dashed) { Set-CellFormula $line "LinePattern" "2" }
    if (-not [string]::IsNullOrWhiteSpace($Label)) {
        $line.Text = Convert-UnicodeEscapes $Label
        Set-CellFormula $line "Char.Font" 'FONT("Microsoft YaHei")'
        Set-CellFormula $line "Char.Size" "6.5 pt"
    }
    return $line
}

function Add-Segment {
    param(
        $Page,
        [double]$PageHeight,
        [double]$X1,
        [double]$Y1,
        [double]$X2,
        [double]$Y2,
        [string]$Color = "#1f4e79",
        [bool]$Dashed = $false,
        [bool]$Arrow = $false
    )
    $line = $Page.DrawLine($X1, ($PageHeight - $Y1), $X2, ($PageHeight - $Y2))
    Set-CellFormula $line "LineColor" (Rgb-Formula $Color)
    Set-CellFormula $line "LineWeight" "0.95 pt"
    if ($Dashed) { Set-CellFormula $line "LinePattern" "2" }
    if ($Arrow) { Set-CellFormula $line "EndArrow" "4" }
    return $line
}

function Add-TextLabel {
    param($Page, [double]$PageHeight, [double]$X, [double]$Y, [double]$W, [double]$H, [string]$Text)
    $def = New-ShapeDef ("label_" + [guid]::NewGuid().ToString("N")) $X $Y $W $H $Text "note" "#ffffff"
    return Add-JournalShape $Page $PageHeight $def
}

$visio = $null
$doc = $null
try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    $visio.AlertResponse = 7
    $doc = $visio.Documents.Add("")

    # Page 1: compact two-panel academic figure.
    $pageWidth = 12.0
    $pageHeight = 8.4
    $page = $visio.ActivePage
    $page.Name = Convert-UnicodeEscapes "\u603b\u89c8_\u6a21\u5757C\u5b66\u672f\u6d41\u7a0b"
    $page.PageSheet.CellsU("PageWidth").ResultIU = $pageWidth
    $page.PageSheet.CellsU("PageHeight").ResultIU = $pageHeight

    $defs = @(
        New-ShapeDef "title" 0.25 0.22 8.5 0.30 "Dual-D \u591a\u6a21\u6001\u8de8\u57df\u6d77\u4e0a\u76ee\u6807\u8bc6\u522b\u6846\u67b6" "title"
        New-ShapeDef "legend" 9.05 0.25 5.50 0.28 "\u77e9\u5f62=\u5904\u7406  \u83f1\u5f62=\u6761\u4ef6  \u84dd\u5b9e\u7ebf=\u7279\u5f81\u6d41  \u7eff\u5b9e\u7ebf=\u635f\u5931\u6d41  \u6a59\u865a\u7ebf=\u53cd\u5411\u4f20\u64ad" "legend"

        New-ShapeDef "overall_panel" 6.00 1.58 11.60 1.72 "(a)  \u6574\u4f53\u8bad\u7ec3\u6d41\u7a0b" "panel"
        New-ShapeDef "input" 0.88 1.55 1.18 0.72 "\u6210\u5bf9 VIS/IR\n\u6674\u5929/\u6076\u52a3\u5929\u6c14" "terminal" "#eaf2fb"
        New-ShapeDef "module_a" 2.25 1.55 1.38 0.72 "A  \u5171\u4eab\u7f16\u7801\u5668\n+ TAL \u5f20\u91cf\u5bf9\u9f50" "main" "#eaf2fb"
        New-ShapeDef "features_a" 3.55 1.55 0.88 0.72 "F_s, F_t\nL_cls,L_TAL" "main" "#edf6f0"
        New-ShapeDef "module_b" 5.02 1.55 1.67 0.72 "B  G_t2s/G_s2t\n+ D_s/D_t \u53cc\u5224\u522b\u5668" "main" "#fff3df"
        New-ShapeDef "features_b" 6.57 1.55 1.02 0.72 "F_ts,F_st\nL_adv^s,L_adv^t" "main" "#fff8eb"
        New-ShapeDef "module_c" 8.25 1.55 2.05 0.78 "C  \u8bed\u4e49\u4fdd\u6301 + \u7c7b\u522b\u611f\u77e5\u53cd\u9988\nL_C={L_id,L_cyc,L_con,\nL_proto,L_gcls}" "main" "#f2edfb"
        New-ShapeDef "module_d" 10.75 1.55 1.95 0.86 "D  \u8054\u5408\u76ee\u6807\nL_total=L_cls+\u03bb_TAL L_TAL+\u03bb_adv L_adv\n+\u03bb_C L_C\uff1b\u4ea4\u66ff\u4f18\u5316 / \u53cd\u5411\u4f20\u64ad" "main" "#eaf4ec"

        New-ShapeDef "c_panel" 6.00 5.18 11.60 5.08 "(b)  \u6a21\u5757 C\uff1a\u8bed\u4e49\u4fdd\u6301\u4e0e\u7c7b\u522b\u611f\u77e5\u53cd\u9988\uff08\u8bad\u7ec3\u671f\uff09" "panel"
        New-ShapeDef "lane_c1" 6.00 3.35 11.34 0.76 "C1  Identity \u8eab\u4efd\u4fdd\u6301" "lane" "#f7fafc"
        New-ShapeDef "lane_c2" 6.00 4.22 11.34 0.76 "C2  Cycle \u53cc\u5411\u95ed\u73af" "lane" "#f7fafc"
        New-ShapeDef "lane_c3" 6.00 5.35 11.34 1.34 "C3  \u914d\u5bf9 / \u7c7b\u522b\u539f\u578b\u5bf9\u6bd4" "lane" "#faf8fd"
        New-ShapeDef "lane_c4" 6.00 6.62 11.34 1.02 "C4  \u751f\u6210\u7279\u5f81\u5206\u7c7b\u53cd\u9988" "lane" "#f7fafc"

        New-ShapeDef "c1_in" 1.43 3.40 1.18 0.50 "F_s, F_t" "terminal" "#eaf2fb"
        New-ShapeDef "c1_map" 3.30 3.40 2.00 0.56 "\u540c\u57df\u6052\u7b49\u6620\u5c04\nF_s^id=G_t2s(F_s)\nF_t^id=G_s2t(F_t)" "process" "#eef7ed"
        New-ShapeDef "c1_calc" 5.85 3.40 2.55 0.56 "L_id=||F_s^id-F_s||_1\n     +||F_t^id-F_t||_1" "process" "#eef7ed"
        New-ShapeDef "c1_effect" 8.55 3.40 1.62 0.52 "\u4fdd\u62a4\u5f31\u8f6e\u5ed3\n\u6291\u5236\u65e0\u8c13\u6539\u5199" "effect" "#f4eefb"
        New-ShapeDef "c1_loss" 10.35 3.40 0.86 0.48 "L_id" "loss"

        New-ShapeDef "c2_in" 1.43 4.27 1.36 0.50 "F_ts,F_st\n+ F_t,F_s" "terminal" "#eaf2fb"
        New-ShapeDef "c2_map" 3.30 4.27 2.00 0.56 "\u9006\u5411\u91cd\u8bd1\nF_t_hat=G_s2t(F_ts)\nF_s_hat=G_t2s(F_st)" "process" "#eef7ed"
        New-ShapeDef "c2_calc" 5.85 4.27 2.55 0.56 "L_cyc=||F_s_hat-F_s||_1\n       +||F_t_hat-F_t||_1" "process" "#eef7ed"
        New-ShapeDef "c2_effect" 8.55 4.27 1.62 0.52 "\u4fdd\u8bc1\u7ffb\u8bd1\u53ef\u9006\n\u9632\u6b62\u7ed3\u6784\u4e22\u5931" "effect" "#f4eefb"
        New-ShapeDef "c2_loss" 10.35 4.27 0.86 0.48 "L_cyc" "loss"

        New-ShapeDef "c3_in" 1.43 5.18 1.44 0.54 "F_s,F_t,F_ts,F_st\n y_s,y_t" "terminal" "#eaf2fb"
        New-ShapeDef "c3_pair" 3.35 5.18 2.12 0.72 "\u914d\u5bf9\u5bf9\u6bd4 PCE\nL_con=1/2[PCE(F_ts,F_s;y)+PCE(F_st,F_t;y)]\ny\u53ef\u7528\uff1a\u540c\u7c7b\u4e3a\u6b63\u6837\u672c\uff1b\u5426\u5219\uff1a\u5bf9\u89d2\u914d\u5bf9" "process" "#f4eefb"
        New-ShapeDef "c3_decision" 6.25 5.18 1.20 0.78 "y_s,y_t\n\u5747\u53ef\u7528\uff1f" "decision"
        New-ShapeDef "c3_proto" 8.28 5.18 2.55 0.72 "P_s^c=mean(F_s|y_s=c), P_t^c=mean(F_t|y_t=c)\nL_proto=1/2[CE(sim(F_ts,P_s)/\u03c4,y_t)\n                 +CE(sim(F_st,P_t)/\u03c4,y_s)]" "process" "#f4eefb"
        New-ShapeDef "c3_skip" 8.28 5.76 1.44 0.36 "L_proto=0" "process" "#f8f9fb"
        New-ShapeDef "c3_loss" 10.35 5.18 1.02 0.56 "L_con\nL_proto" "loss"

        New-ShapeDef "c4_in" 1.43 6.48 1.26 0.50 "F_ts,F_st" "terminal" "#eaf2fb"
        New-ShapeDef "c4_decision" 3.35 6.48 1.32 0.80 "\u5206\u7c7b\u5668 C\n\u4e0e CE \u53ef\u7528\uff1f" "decision"
        New-ShapeDef "c4_calc" 5.85 6.48 2.90 0.66 "z_ts=C(F_ts), z_st=C(F_st)\nL_gcls=1[y_t]CE(z_ts,y_t)+1[y_s]CE(z_st,y_s)" "process" "#eef7ed"
        New-ShapeDef "c4_skip" 5.85 7.02 1.42 0.34 "L_gcls=0" "process" "#f8f9fb"
        New-ShapeDef "c4_effect" 8.55 6.48 1.62 0.52 "\u7ffb\u8bd1\u540e\u4ecd\u53ef\u5206\n\u4fdd\u62a4\u4efb\u52a1\u8bed\u4e49" "effect" "#f4eefb"
        New-ShapeDef "c4_loss" 10.35 6.48 0.90 0.48 "L_gcls" "loss"

        New-ShapeDef "c_aggregate" 11.36 4.94 0.64 3.75 "\u635f\u5931\n\u6c47\u603b\n\u2192 D" "aggregate" "#ddeee0"
        New-ShapeDef "bp_note" 4.70 7.48 7.60 0.24 "\u53cd\u5411\u4f20\u64ad\uff1aL_total \u66f4\u65b0\u5171\u4eab\u7f16\u7801\u5668\u3001TAL\u3001G_t2s/G_s2t \u4e0e\u5206\u7c7b\u5668 C" "note"
    )

    $shapeMap = @{}
    foreach ($def in $defs) {
        $shapeMap[[string]$def.key] = Add-JournalShape $page $pageHeight $def
    }

    # Overall forward path.
    Add-Arrow $page $shapeMap "input" "module_a" | Out-Null
    Add-Arrow $page $shapeMap "module_a" "features_a" | Out-Null
    Add-Arrow $page $shapeMap "features_a" "module_b" | Out-Null
    Add-Arrow $page $shapeMap "module_b" "features_b" | Out-Null
    Add-Arrow $page $shapeMap "features_b" "module_c" | Out-Null

    # Loss bus in the overall panel.
    Add-Segment $page $pageHeight 3.55 1.92 3.55 2.20 "#2e7d32" $false $false | Out-Null
    Add-Segment $page $pageHeight 6.57 1.92 6.57 2.20 "#2e7d32" $false $false | Out-Null
    Add-Segment $page $pageHeight 8.25 1.94 8.25 2.20 "#2e7d32" $false $false | Out-Null
    Add-Segment $page $pageHeight 3.55 2.20 10.75 2.20 "#2e7d32" $false $false | Out-Null
    Add-Segment $page $pageHeight 10.75 2.20 10.75 1.98 "#2e7d32" $false $true | Out-Null
    Add-TextLabel $page $pageHeight 7.15 2.28 4.60 0.18 "L_cls + L_TAL + L_adv + L_C" | Out-Null

    # C1/C2 forward computation and loss outputs.
    foreach ($row in @(
        @("c1_in","c1_map","c1_calc","c1_effect","c1_loss"),
        @("c2_in","c2_map","c2_calc","c2_effect","c2_loss")
    )) {
        Add-Arrow $page $shapeMap $row[0] $row[1] | Out-Null
        Add-Arrow $page $shapeMap $row[1] $row[2] | Out-Null
        Add-Arrow $page $shapeMap $row[2] $row[3] | Out-Null
        Add-Arrow $page $shapeMap $row[3] $row[4] "E" "W" "#2e7d32" | Out-Null
    }

    # C3: paired contrast is always computed; prototype feedback is conditional.
    Add-Arrow $page $shapeMap "c3_in" "c3_pair" | Out-Null
    Add-Arrow $page $shapeMap "c3_pair" "c3_decision" | Out-Null
    Add-Arrow $page $shapeMap "c3_decision" "c3_proto" "E" "W" "#1f4e79" $false "\u662f" | Out-Null
    Add-Arrow $page $shapeMap "c3_proto" "c3_loss" "E" "W" "#2e7d32" | Out-Null
    Add-Segment $page $pageHeight 6.25 5.57 6.25 5.76 "#1f4e79" $false $false | Out-Null
    Add-Segment $page $pageHeight 6.25 5.76 7.56 5.76 "#1f4e79" $false $true | Out-Null
    Add-TextLabel $page $pageHeight 6.55 5.67 0.38 0.16 "\u5426" | Out-Null
    Add-Segment $page $pageHeight 9.00 5.76 9.73 5.76 "#2e7d32" $false $false | Out-Null
    Add-Segment $page $pageHeight 9.73 5.76 9.73 5.18 "#2e7d32" $false $false | Out-Null
    Add-Segment $page $pageHeight 9.73 5.18 9.84 5.18 "#2e7d32" $false $true | Out-Null

    # C4: classification feedback is enabled only when the required supervision exists.
    Add-Arrow $page $shapeMap "c4_in" "c4_decision" | Out-Null
    Add-Arrow $page $shapeMap "c4_decision" "c4_calc" "E" "W" "#1f4e79" $false "\u662f" | Out-Null
    Add-Arrow $page $shapeMap "c4_calc" "c4_effect" | Out-Null
    Add-Arrow $page $shapeMap "c4_effect" "c4_loss" "E" "W" "#2e7d32" | Out-Null
    Add-Segment $page $pageHeight 3.35 6.88 3.35 7.02 "#1f4e79" $false $false | Out-Null
    Add-Segment $page $pageHeight 3.35 7.02 5.14 7.02 "#1f4e79" $false $true | Out-Null
    Add-TextLabel $page $pageHeight 3.68 6.94 0.38 0.16 "\u5426" | Out-Null
    Add-Segment $page $pageHeight 6.56 7.02 9.74 7.02 "#2e7d32" $false $false | Out-Null
    Add-Segment $page $pageHeight 9.74 7.02 9.74 6.48 "#2e7d32" $false $false | Out-Null
    Add-Segment $page $pageHeight 9.74 6.48 9.90 6.48 "#2e7d32" $false $true | Out-Null

    # Four outputs enter the collector at their own row heights, avoiding fan-in diagonals.
    Add-Segment $page $pageHeight 10.78 3.40 11.04 3.40 "#2e7d32" $false $true | Out-Null
    Add-Segment $page $pageHeight 10.78 4.27 11.04 4.27 "#2e7d32" $false $true | Out-Null
    Add-Segment $page $pageHeight 10.86 5.18 11.04 5.18 "#2e7d32" $false $true | Out-Null
    Add-Segment $page $pageHeight 10.80 6.48 11.04 6.48 "#2e7d32" $false $true | Out-Null

    # Module-C loss bundle enters D using a short orthogonal route.
    Add-Segment $page $pageHeight 11.36 3.06 11.36 2.44 "#2e7d32" $false $false | Out-Null
    Add-Segment $page $pageHeight 11.36 2.44 10.75 2.44 "#2e7d32" $false $false | Out-Null
    Add-Segment $page $pageHeight 10.75 2.44 10.75 1.98 "#2e7d32" $false $true | Out-Null

    # Backpropagation stays outside the feature lanes.
    Add-Segment $page $pageHeight 11.36 6.82 11.72 6.82 "#c55a11" $true $false | Out-Null
    Add-Segment $page $pageHeight 11.72 6.82 11.72 7.60 "#c55a11" $true $false | Out-Null
    Add-Segment $page $pageHeight 11.72 7.60 0.78 7.60 "#c55a11" $true $true | Out-Null

    # Page 2: publication-ready formula and symbol reference, separated from the architecture.
    $formulaWidth = 8.3
    $formulaHeight = 11.7
    $formulaPage = $doc.Pages.Add()
    $formulaPage.Name = Convert-UnicodeEscapes "\u635f\u5931\u51fd\u6570_\u516c\u5f0f\u4e0e\u7b26\u53f7"
    $formulaPage.PageSheet.CellsU("PageWidth").ResultIU = $formulaWidth
    $formulaPage.PageSheet.CellsU("PageHeight").ResultIU = $formulaHeight

    $formulaDefs = @(
        New-ShapeDef "f_title" 0.35 0.35 7.3 0.38 "Dual-D \u635f\u5931\u51fd\u6570\u4e0e\u7b26\u53f7\u8bf4\u660e" "title"
        New-ShapeDef "f_total" 4.15 1.20 7.55 1.05 "\u8054\u5408\u76ee\u6807\nL_total = L_cls + \u03bb_TAL L_TAL + \u03bb_s L_adv^s + \u03bb_t L_adv^t + \u03bb_cyc L_cyc\n        + \u03bb_id L_id + \u03bb_con L_con + \u03bb_proto L_proto + \u03bb_gcls L_gcls" "formula" "#eaf4ec"
        New-ShapeDef "f_cls" 4.15 2.15 7.55 0.62 "L_cls = CE(C(F_s),y_s) + CE(C(F_t),y_t)    \u2014    \u76d1\u7763\u4efb\u52a1\u5206\u7c7b" "formula" "#f8fafc"
        New-ShapeDef "f_tal" 4.15 2.90 7.55 0.62 "L_TAL = ||A_s-A_t||_2^2    \u2014    \u51cf\u5c0f\u591a\u6a21\u6001/\u8de8\u57df\u5f20\u91cf\u5dee\u5f02" "formula" "#f8fafc"
        New-ShapeDef "f_adv" 4.15 3.78 7.55 0.86 "L_adv^s=CE(D_s(F_ts),1),  L_adv^t=CE(D_t(F_st),1)\nL_D=1/2(L_D^s+L_D^t),  L_D^k=1/2[CE(D_k(F_real),1)+CE(D_k(F_fake),0)]" "formula" "#fff8eb"
        New-ShapeDef "f_id" 4.15 4.82 7.55 0.72 "L_id = ||G_t2s(F_s)-F_s||_1 + ||G_s2t(F_t)-F_t||_1\n\u751f\u6210\u5668\u9047\u5230\u5df2\u5c5e\u4e8e\u8f93\u51fa\u57df\u7684\u7279\u5f81\u65f6\u4fdd\u6301\u8fd1\u4f3c\u6052\u7b49\u6620\u5c04\u3002" "formula" "#eef7ed"
        New-ShapeDef "f_cyc" 4.15 5.75 7.55 0.82 "F_t_hat=G_s2t(F_ts),  F_s_hat=G_t2s(F_st)\nL_cyc=||F_s_hat-F_s||_1+||F_t_hat-F_t||_1    \u2014    \u7ea6\u675f\u7ffb\u8bd1\u53ef\u9006\u6027" "formula" "#eef7ed"
        New-ShapeDef "f_con" 4.15 6.72 7.55 0.82 "L_con=1/2[PCE(F_ts,F_s;y)+PCE(F_st,F_t;y)]\nPCE(a,p;y)=-E_i[|P_i|^{-1}\u2211_{j\u2208P_i} log exp(sim(a_i,p_j)/\u03c4)/\u2211_k exp(sim(a_i,p_k)/\u03c4)]" "formula" "#f4eefb"
        New-ShapeDef "f_proto" 4.15 7.92 7.55 1.18 "P_s^c=mean_{i:y_s_i=c}(F_s_i),  P_t^c=mean_{i:y_t_i=c}(F_t_i)\nL_proto=1/2[CE(sim(F_ts,P_s)/\u03c4,y_t)+CE(sim(F_st,P_t)/\u03c4,y_s)]\n\u6bcf\u6279\u4ec5\u5bf9\u5b58\u5728\u7684\u7c7b\u522b\u539f\u578b\u6c42\u635f\u5931\uff1b\u7f3a\u5c11\u53cc\u57df\u6807\u7b7e\u65f6 L_proto=0\u3002" "formula" "#f4eefb"
        New-ShapeDef "f_gcls" 4.15 9.12 7.55 0.82 "z_ts=C(F_ts),  z_st=C(F_st)\nL_gcls=1[y_t]CE(z_ts,y_t)+1[y_s]CE(z_st,y_s)    \u2014    \u4ec5\u5bf9\u6807\u7b7e\u53ef\u7528\u7684\u65b9\u5411\u8ba1\u7b97" "formula" "#eef7ed"
        New-ShapeDef "f_symbols" 4.15 10.45 7.55 1.30 "\u7b26\u53f7\uff1aF_s/F_t=\u6e90/\u76ee\u6807\u57df\u771f\u5b9e\u7279\u5f81\uff1bF_ts=G_t2s(F_t)\uff1bF_st=G_s2t(F_s)\uff1b\nP_s^c/P_t^c=\u7c7b c \u7684\u6e90/\u76ee\u6807\u57df\u539f\u578b\uff1by_s/y_t=\u6837\u672c\u7c7b\u522b\uff1b\u03c4=\u5bf9\u6bd4\u6e29\u5ea6\uff1b1[y]=\u6807\u7b7e\u53ef\u7528\u6307\u793a\u51fd\u6570\u3002\n\u8bad\u7ec3\uff1a\u5224\u522b\u5668\u4e0e\u751f\u6210/\u8bc6\u522b\u7f51\u7edc\u4ea4\u66ff\u66f4\u65b0\uff1b\u5bf9\u6297\u6743\u91cd\u91c7\u7528\u9884\u70ed\u4e0e\u6e10\u589e\u3002" "formula" "#f8fafc"
    )
    foreach ($def in $formulaDefs) {
        Add-JournalShape $formulaPage $formulaHeight $def | Out-Null
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
        $formulaPage.Export($previewBase + "_formulas" + $previewExtension)
        Write-Output "Exported $PreviewPath"
    }
} finally {
    if ($doc -ne $null) { $doc.Close() }
    if ($visio -ne $null) { $visio.Quit() }
}
