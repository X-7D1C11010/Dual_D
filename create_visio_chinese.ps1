$ErrorActionPreference = 'Stop'

$outVsdx = Join-Path $PSScriptRoot 'image-gen-2_中文版.vsdx'
$outPng = Join-Path $PSScriptRoot 'image-gen-2_中文版_预览.png'

function Set-Cell {
    param($Shape, [string]$Cell, [string]$Formula)
    try { $Shape.CellsU($Cell).FormulaU = $Formula } catch { }
}

function Add-Rect {
    param(
        $Page, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2,
        [string]$Text = '', [int[]]$Fill = @(255,255,255), [int[]]$Line = @(70,80,90),
        [double]$FontSize = 8.0, [bool]$Bold = $false, [double]$Radius = 0.08,
        [int]$Align = 1, [int]$Transparency = 0
    )
    $s = $Page.DrawRectangle($X1,$Y1,$X2,$Y2)
    $s.Text = $Text
    Set-Cell $s 'FillForegnd' ("RGB({0},{1},{2})" -f $Fill[0],$Fill[1],$Fill[2])
    Set-Cell $s 'FillPattern' '1'
    Set-Cell $s 'FillTransparency' ("{0}%" -f $Transparency)
    Set-Cell $s 'LineColor' ("RGB({0},{1},{2})" -f $Line[0],$Line[1],$Line[2])
    Set-Cell $s 'LineWeight' '0.65 pt'
    Set-Cell $s 'Rounding' ("{0} in" -f $Radius)
    Set-Cell $s 'Char.Size' ("{0} pt" -f $FontSize)
    Set-Cell $s 'Char.Style' ($(if ($Bold) { '1' } else { '0' }))
    Set-Cell $s 'Para.HorzAlign' ([string]$Align)
    Set-Cell $s 'VerticalAlign' '1'
    Set-Cell $s 'TxtMarginLeft' '0.04 in'
    Set-Cell $s 'TxtMarginRight' '0.04 in'
    Set-Cell $s 'TxtMarginTop' '0.02 in'
    Set-Cell $s 'TxtMarginBottom' '0.02 in'
    return $s
}

function Add-Text {
    param($Page, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2, [string]$Text,
          [double]$FontSize = 9.0, [bool]$Bold = $false, [int[]]$Color = @(30,35,40), [int]$Align = 1)
    $s = Add-Rect $Page $X1 $Y1 $X2 $Y2 $Text @(255,255,255) @(255,255,255) $FontSize $Bold 0 $Align 100
    Set-Cell $s 'LinePattern' '0'
    Set-Cell $s 'Char.Color' ("RGB({0},{1},{2})" -f $Color[0],$Color[1],$Color[2])
    return $s
}

function Add-Line {
    param($Page, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2, [bool]$Arrow = $false, [bool]$Dashed = $false)
    $s = $Page.DrawLine($X1,$Y1,$X2,$Y2)
    Set-Cell $s 'LineColor' 'RGB(25,30,35)'
    Set-Cell $s 'LineWeight' '0.75 pt'
    if ($Arrow) { Set-Cell $s 'EndArrow' '13' }
    if ($Dashed) { Set-Cell $s 'LinePattern' '2' }
    return $s
}

function Add-Arrow {
    param($Page, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2)
    return Add-Line $Page $X1 $Y1 $X2 $Y2 $true $false
}

function Add-ElbowArrow {
    param($Page, [double]$X1, [double]$Y1, [double]$Xm, [double]$Ym, [double]$X2, [double]$Y2)
    [void](Add-Line $Page $X1 $Y1 $Xm $Y1 $false $false)
    [void](Add-Line $Page $Xm $Y1 $Xm $Ym $false $false)
    [void](Add-Arrow $Page $Xm $Ym $X2 $Y2)
}

function Add-Matrix {
    param($Page, [double]$Left, [double]$Bottom, [string]$Theme)
    $blue = @(@(220,235,249),@(139,190,229),@(56,126,184),@(23,83,145))
    $green = @(@(226,241,220),@(154,205,143),@(72,157,82),@(28,112,55))
    $pal = if ($Theme -eq 'blue') { $blue } else { $green }
    $patterns = @(
        @(0,1,2,1),
        @(1,3,1,2),
        @(2,1,3,1),
        @(1,2,1,0)
    )
    $cell = 0.105
    for ($r=0; $r -lt 4; $r++) {
        for ($c=0; $c -lt 4; $c++) {
            $idx = $patterns[$r][$c]
            $x1 = $Left + $c*$cell
            $y1 = $Bottom + (3-$r)*$cell
            $s = Add-Rect $Page $x1 $y1 ($x1+$cell) ($y1+$cell) '' $pal[$idx] @(190,200,205) 1 $false 0 1 0
            Set-Cell $s 'LineWeight' '0.25 pt'
        }
    }
}

function Add-FlowRow {
    param($Page, [double]$YBottom, [string]$Dir, [string]$Kind)

    $isTop = $Kind -eq 'reference'
    $blueFill = @(226,240,252)
    $greenFill = @(232,246,226)
    $white = @(255,255,255)

    if ($Dir -eq 'st') {
        if ($isTop) {
            $leftTitle = '参考融合特征'
            $leftFormula = "Zₛ = [ Zₛ^(SAR) ; Zₛ^(Opt) ]"
            $tag = '停止梯度参考'
            $sub = 's'
            $arrow = 's→t'
        } else {
            $leftTitle = '翻译特征'
            $leftFormula = "Ẑₛ→ₜ = [ Ẑₛ→ₜ^(SAR) ; Ẑₛ→ₜ^(Opt) ]"
            $tag = ''
            $sub = 's→t'
            $arrow = 's→t'
        }
    } else {
        if ($isTop) {
            $leftTitle = '参考融合特征'
            $leftFormula = "Zₜ = [ Zₜ^(SAR) ; Zₜ^(Opt) ]"
            $tag = '停止梯度参考'
            $sub = 't'
            $arrow = 't→s'
        } else {
            $leftTitle = '翻译特征'
            $leftFormula = "Ẑₜ→ₛ = [ Ẑₜ→ₛ^(SAR) ; Ẑₜ→ₛ^(Opt) ]"
            $tag = ''
            $sub = 't→s'
            $arrow = 't→s'
        }
    }

    # Fused feature block
    [void](Add-Rect $Page 0.30 $YBottom 2.45 ($YBottom+0.92) ("{0}`n{1}" -f $leftTitle,$leftFormula) $white @(70,80,90) 8.0 $false 0.08 1 0)
    if ($tag) {
        [void](Add-Rect $Page 0.48 ($YBottom-0.08) 2.25 ($YBottom+0.15) $tag @(224,231,238) @(224,231,238) 6.6 $false 0.05 1 0)
    }

    # Split by modality
    [void](Add-Rect $Page 2.65 ($YBottom-0.03) 4.78 ($YBottom+1.02) '' $white @(70,80,90) 7 $false 0.07 1 0)
    [void](Add-Text $Page 2.67 ($YBottom+0.79) 4.76 ($YBottom+1.00) '按模态拆分' 8.2 $true @(25,30,35) 1)
    $sarText = if ($isTop) { "SAR 模态块   Z$sub^(SAR)   (N × d)" } else { "SAR 模态块   Ẑ$sub^(SAR)   (N × d)" }
    $optText = if ($isTop) { "光学模态块   Z$sub^(Opt)   (N × d)" } else { "光学模态块   Ẑ$sub^(Opt)   (N × d)" }
    [void](Add-Rect $Page 2.73 ($YBottom+0.45) 4.70 ($YBottom+0.76) $sarText $blueFill $blueFill 7.0 $false 0.05 1 0)
    [void](Add-Rect $Page 2.73 ($YBottom+0.08) 4.70 ($YBottom+0.39) $optText $greenFill $greenFill 7.0 $false 0.05 1 0)

    # L2 normalization
    [void](Add-Rect $Page 5.00 ($YBottom-0.03) 6.95 ($YBottom+1.02) '' $white @(70,80,90) 7 $false 0.07 1 0)
    [void](Add-Text $Page 5.02 ($YBottom+0.79) 6.93 ($YBottom+1.00) 'L2 归一化' 8.2 $true @(25,30,35) 1)
    if ($isTop) {
        $sarNorm = "Z̄$sub^(SAR) = Z$sub^(SAR) / ‖Z$sub^(SAR)‖₂"
        $optNorm = "Z̄$sub^(Opt) = Z$sub^(Opt) / ‖Z$sub^(Opt)‖₂"
    } else {
        $sarNorm = "Z̄$sub^(SAR) = Ẑ$sub^(SAR) / ‖Ẑ$sub^(SAR)‖₂"
        $optNorm = "Z̄$sub^(Opt) = Ẑ$sub^(Opt) / ‖Ẑ$sub^(Opt)‖₂"
    }
    [void](Add-Rect $Page 5.07 ($YBottom+0.45) 6.88 ($YBottom+0.76) $sarNorm $blueFill $blueFill 6.6 $false 0.04 1 0)
    [void](Add-Rect $Page 5.07 ($YBottom+0.08) 6.88 ($YBottom+0.39) $optNorm $greenFill $greenFill 6.6 $false 0.04 1 0)

    # Relation matrices
    [void](Add-Rect $Page 7.20 ($YBottom-0.03) 10.10 ($YBottom+1.02) '' $white @(70,80,90) 7 $false 0.07 1 0)
    [void](Add-Text $Page 7.22 ($YBottom+0.79) 10.08 ($YBottom+1.00) '关系矩阵' 8.2 $true @(25,30,35) 1)
    [void](Add-Text $Page 7.33 ($YBottom+0.48) 8.73 ($YBottom+0.76) ("K$sub^(SAR) = Z̄$sub^(SAR)(Z̄$sub^(SAR))ᵀ") 6.3 $false @(30,35,40) 0)
    [void](Add-Text $Page 7.33 ($YBottom+0.10) 8.73 ($YBottom+0.38) ("K$sub^(Opt) = Z̄$sub^(Opt)(Z̄$sub^(Opt))ᵀ") 6.3 $false @(30,35,40) 0)
    Add-Matrix $Page 8.85 ($YBottom+0.38) 'blue'
    Add-Matrix $Page 8.85 $YBottom 'green'
    [void](Add-Text $Page 9.35 ($YBottom+0.45) 10.02 ($YBottom+0.74) ("K$sub^(SAR)`n(N × N)") 6.1 $false @(30,35,40) 1)
    [void](Add-Text $Page 9.35 ($YBottom+0.07) 10.02 ($YBottom+0.36) ("K$sub^(Opt)`n(N × N)") 6.1 $false @(30,35,40) 1)

    # Modality inconsistency
    [void](Add-Rect $Page 10.35 ($YBottom-0.03) 12.70 ($YBottom+1.02) '' $white @(70,80,90) 7 $false 0.07 1 0)
    if ($isTop) { [void](Add-Text $Page 10.37 ($YBottom+0.79) 12.68 ($YBottom+1.00) '模态不一致度' 8.2 $true @(25,30,35) 1) }
    $a = if ($isTop) { "A(Z$sub) = ‖K$sub^(SAR) − K$sub^(Opt)‖²_F" } else { "A(Ẑ$sub) = ‖K$sub^(SAR) − K$sub^(Opt)‖²_F" }
    [void](Add-Text $Page 10.49 ($YBottom+0.39) 12.55 ($YBottom+0.70) $a 7.0 $false @(30,35,40) 1)
    $aTag = if ($isTop) { "A(Z$sub)" } else { "A(Ẑ$sub)" }
    [void](Add-Rect $Page 10.95 ($YBottom+0.09) 12.08 ($YBottom+0.34) $aTag @(229,240,251) @(229,240,251) 7.2 $false 0.04 1 0)

    # Flow arrows
    [void](Add-Arrow $Page 2.45 ($YBottom+0.48) 2.65 ($YBottom+0.48))
    [void](Add-Arrow $Page 4.78 ($YBottom+0.60) 5.00 ($YBottom+0.60))
    [void](Add-Arrow $Page 4.78 ($YBottom+0.23) 5.00 ($YBottom+0.23))
    [void](Add-Arrow $Page 6.95 ($YBottom+0.60) 7.20 ($YBottom+0.60))
    [void](Add-Arrow $Page 6.95 ($YBottom+0.23) 7.20 ($YBottom+0.23))
    [void](Add-Arrow $Page 10.10 ($YBottom+0.48) 10.35 ($YBottom+0.48))
}

$visio = $null
$doc = $null
try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    $doc = $visio.Documents.Add('')
    $page = $visio.ActivePage
    $page.Name = '双向关系漂移约束'
    Set-Cell $page.PageSheet 'PageWidth' '16.7 in'
    Set-Cell $page.PageSheet 'PageHeight' '9.4 in'
    Set-Cell $page.PageSheet 'ShdwPattern' '0'

    # Background path containers
    $topBg = Add-Rect $page 0.12 5.62 16.58 9.30 '' @(246,251,255) @(121,183,232) 7 $false 0.10 1 35
    Set-Cell $topBg 'LinePattern' '2'
    Set-Cell $topBg 'LineWeight' '0.8 pt'
    $botBg = Add-Rect $page 0.12 1.75 16.58 5.48 '' @(248,253,248) @(129,197,142) 7 $false 0.10 1 35
    Set-Cell $botBg 'LinePattern' '2'
    Set-Cell $botBg 'LineWeight' '0.8 pt'

    [void](Add-Text $page 0.22 9.00 3.25 9.30 '(A) 源到目标路径' 13.5 $true @(44,94,154) 0)
    [void](Add-Text $page 0.22 5.18 3.25 5.46 '(B) 目标到源路径' 13.5 $true @(43,112,60) 0)

    Add-FlowRow $page 7.63 'st' 'reference'
    Add-FlowRow $page 6.10 'st' 'translated'
    Add-FlowRow $page 3.80 'ts' 'reference'
    Add-FlowRow $page 2.27 'ts' 'translated'

    # Top drift and hinge boxes
    [void](Add-Rect $page 13.08 7.00 14.55 8.35 "关系漂移（比较）`n`nΔₛ→ₜ =`nA(Ẑₛ→ₜ) − A(Zₛ)" @(255,244,239) @(195,93,61) 7.5 $true 0.08 1 0)
    [void](Add-Rect $page 14.75 7.00 16.40 8.35 "铰链 / 容差`n`nDₛ→ₜ =`nmax(0, Δₛ→ₜ − τ)" @(251,245,255) @(138,81,157) 7.5 $true 0.08 1 0)
    [void](Add-Arrow $page 12.70 8.06 13.08 8.06)
    [void](Add-ElbowArrow $page 12.70 6.58 12.90 7.30 13.08 7.30)
    [void](Add-Arrow $page 14.55 7.68 14.75 7.68)

    # Bottom drift and hinge boxes
    [void](Add-Rect $page 13.08 3.17 14.55 4.52 "关系漂移（比较）`n`nΔₜ→ₛ =`nA(Ẑₜ→ₛ) − A(Zₜ)" @(255,244,239) @(195,93,61) 7.5 $true 0.08 1 0)
    [void](Add-Rect $page 14.75 3.17 16.40 4.52 "铰链 / 容差`n`nDₜ→ₛ =`nmax(0, Δₜ→ₛ − τ)" @(251,245,255) @(138,81,157) 7.5 $true 0.08 1 0)
    [void](Add-Arrow $page 12.70 4.23 13.08 4.23)
    [void](Add-ElbowArrow $page 12.70 2.75 12.90 3.47 13.08 3.47)
    [void](Add-Arrow $page 14.55 3.85 14.75 3.85)

    # Combined loss and schedule
    [void](Add-Rect $page 5.92 0.28 10.72 1.28 "关系漂移损失`nℒ_drift = 1/2 · (Dₛ→ₜ + Dₜ→ₛ)`n仅惩罚超出容差 τ 的关系退化。" @(255,239,239) @(190,48,48) 9.0 $false 0.10 1 0)
    [void](Add-Rect $page 11.75 0.45 14.15 1.12 "预热与渐增调度" @(250,250,250) @(130,130,130) 8.0 $false 0.06 1 0)
    Set-Cell $page.Shapes.Item($page.Shapes.Count) 'LinePattern' '2'
    [void](Add-Line $page 11.75 0.78 10.72 0.78 $true $true)

    # Hinge outputs to the final loss
    [void](Add-Line $page 16.40 7.68 16.55 7.68 $false $false)
    [void](Add-Line $page 16.55 7.68 16.55 1.53 $false $false)
    [void](Add-Line $page 16.55 1.53 8.32 1.53 $false $false)
    [void](Add-Arrow $page 8.32 1.53 8.32 1.28)
    [void](Add-Line $page 15.58 3.17 15.58 1.53 $false $false)

    # Keep backgrounds behind all foreground shapes.
    $topBg.SendToBack()
    $botBg.SendToBack()

    $doc.SaveAs($outVsdx)
    $page.Export($outPng)
    $doc.Close()
    $visio.Quit()
    Write-Output "Created: $outVsdx"
    Write-Output "Preview: $outPng"
}
catch {
    if ($doc) { try { $doc.Close() } catch { } }
    if ($visio) { try { $visio.Quit() } catch { } }
    throw
}
finally {
    if ($doc) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc) }
    if ($visio) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($visio) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
