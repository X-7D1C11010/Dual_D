from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(r"D:\Code\Dual_D")
SOURCE = ROOT / "用于恶劣天气下异构光学与AIS数据的海上目标识别的双对抗网络.docx"
OUTPUT = ROOT / "用于恶劣天气下异构光学与AIS数据的海上目标识别的双对抗网络_方法修订稿.docx"


def set_run_font(run, *, east_asia: str = "宋体", latin: str = "Times New Roman", size: float = 10.5):
    run.font.name = latin
    run.font.size = Pt(size)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.color.rgb = None


def format_body(paragraph):
    paragraph.paragraph_format.first_line_indent = Pt(21)
    paragraph.paragraph_format.line_spacing = Pt(16)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.widow_control = True
    for run in paragraph.runs:
        set_run_font(run)


def format_heading(paragraph, level: int):
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(6 if level == 1 else 3)
    paragraph.paragraph_format.space_after = Pt(2)
    for run in paragraph.runs:
        set_run_font(run, east_asia="黑体", size=12 if level == 1 else 10.5)
        run.bold = True


def format_equation(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Pt(0)
    paragraph.paragraph_format.left_indent = Pt(8)
    paragraph.paragraph_format.right_indent = Pt(8)
    paragraph.paragraph_format.line_spacing = Pt(15)
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_together = True
    for run in paragraph.runs:
        set_run_font(run, east_asia="Cambria Math", latin="Cambria Math", size=10)
        run.italic = True


def clear_and_set(paragraph, text: str):
    paragraph.clear()
    run = paragraph.add_run(text)
    format_body(paragraph)
    return run


def replace_startswith(document: Document, prefix: str, text: str):
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            clear_and_set(paragraph, text)
            return
    raise RuntimeError(f"Paragraph not found: {prefix}")


def append_before(document: Document, anchor, text: str, style: str = "Normal"):
    paragraph = document.add_paragraph(style=style)
    paragraph.add_run(text)
    anchor._p.addprevious(paragraph._p)
    if style == "Heading 1":
        format_heading(paragraph, 1)
    elif style == "Heading 2":
        format_heading(paragraph, 2)
    elif style == "Equation":
        format_equation(paragraph)
    else:
        format_body(paragraph)
    return paragraph


def remove_method_section(document: Document):
    paragraphs = document.paragraphs
    start = next(
        p for p in paragraphs if p.text.strip() == "方法" and p.style.name == "Heading 1"
    )
    passed_start = False
    end = None
    for paragraph in paragraphs:
        if paragraph is start:
            passed_start = True
            continue
        if passed_start and paragraph.text.strip() == "实验" and paragraph.style.name == "Heading 1":
            end = paragraph
            break
    if end is None:
        raise RuntimeError("The experiment heading following the method section was not found.")

    current = start._p
    parent = current.getparent()
    while current is not end._p:
        following = current.getnext()
        parent.remove(current)
        current = following
    return end


def revise_cross_section_summary(document: Document):
    abstract = (
        "摘要：海上目标识别在夜间、逆光、雾霾和降雨等恶劣天气下容易受到光学成像退化与域偏移的共同影响。"
        "可见光、红外与AIS观测具有互补性，但三类数据的表征形式和统计分布差异显著；若将模态融合与跨域对齐割裂处理，"
        "既难以利用异质模态之间的高阶关联，也可能在后续域对齐中破坏已经形成的模态关系。为此，本文提出联合模态—域对齐双对抗网络"
        "（Joint Modality-Domain Alignment Network，JMDA-Net）。该方法以模态特定编码器提取可见光、红外和AIS特征，"
        "通过可微的张量收缩与域特定正交投影联合建模跨模态、跨域相关性；随后利用双向残差生成器和两个方向特定判别器，"
        "分别学习目标域到源域以及源域到目标域的非线性映射。为避免对抗映射改变样本类别语义，网络进一步引入循环一致性、恒等保持、"
        "类内对比、指数滑动类别原型和分类器反馈。考虑到双向域映射可能削弱张量模块已建立的模态协同关系，本文以张量对齐输出为固定参照，"
        "对映射前后各模态样本关系矩阵的恶化量施加单侧漂移约束。中国烟台多源海上数据集上的实验表明，JMDA-Net在四种恶劣天气场景下"
        "取得了94.39%的平均识别准确率，验证了该方法在异质多模态跨天气识别中的有效性。"
    )
    clear_and_set(document.paragraphs[1], abstract)

    replace_startswith(
        document,
        "针对上述问题，本文提出",
        "针对上述问题，本文提出联合模态—域对齐双对抗网络（Joint Modality-Domain Alignment Network，JMDA-Net），用于异质光学与AIS数据的跨天气海上目标识别。JMDA-Net首先在张量对齐阶段联合建模模态间高阶依赖与跨域相关性，得到保留模态互补结构的共享表示；随后通过双向残差生成器和两个方向特定判别器补偿线性投影难以覆盖的非线性域差异。对抗映射仅约束域真实性，不能保证样本身份与类别语义，因此网络同时加入循环一致性、恒等保持、类内对比、类别原型和分类器反馈。最后，以张量对齐输出为固定参照度量映射前后的模态关系变化，只惩罚超过容许边界的关系恶化，从而避免全局域对齐反向破坏前一阶段形成的模态协同结构。",
    )
    replace_startswith(
        document,
        "（1）提出一种面向异质光学",
        "（1）提出一种面向异质光学与AIS数据的联合模态—域对齐双对抗框架。该框架按照“高阶关系对齐—非线性双向域映射—语义与模态关系保护”的顺序组织各模块，使后续优化目标与前一阶段建立的结构约束保持一致。",
    )
    replace_startswith(
        document,
        "（2）设计一种基于张量",
        "（2）设计一种计算可行的高阶模态—域对齐机制。该机制利用秩一外积的因子化收缩避免显式构造高维张量，并通过源域、目标域各自的正交投影与类平衡跨域相关性约束，在统一阶段学习模态互补结构和跨域共享表示。",
    )
    replace_startswith(
        document,
        "（3）设计一种最优传输模块",
        "（3）构建双向残差生成器与方向特定双判别器，并以循环一致性、恒等保持、样本级与原型级对比以及分类器反馈约束映射的类别语义；进一步提出单侧模态关系漂移约束，使全局域对齐在允许必要特征变化的同时，不显著削弱张量模块已形成的模态对齐状态。",
    )
    replace_startswith(
        document,
        "针对恶劣天气条件下海上目标识别性能下降的问题",
        "针对恶劣天气条件下光学成像退化和跨域分布差异共同导致的海上目标识别性能下降问题，本文提出联合模态—域对齐双对抗网络JMDA-Net。该网络先以可微张量收缩和域特定正交投影联合学习跨模态、跨域关系，再通过双向残差生成器与两个方向特定判别器补偿非线性域差异。循环一致性、恒等保持、类内对比、类别原型和分类器反馈用于维持映射前后的类别语义；单侧模态关系漂移约束则以张量对齐结果为固定参照，限制全局域映射对既有模态协同结构的破坏。在中国烟台构建的多源海上数据集上，JMDA-Net在四种恶劣天气场景下取得94.39%的平均识别准确率。后续工作将进一步研究缺失模态条件下的动态容错和面向边缘设备的轻量化部署。",
    )


def build_method(document: Document, anchor):
    blocks = [
        ("Heading 1", "方法"),
        ("Heading 2", "问题定义与总体框架"),
        (
            "Normal",
            "本文研究同一类别空间下的跨天气多模态识别。以成像条件稳定的晴天数据为源域，以夜间、逆光、雾霾或降雨数据为目标域。每个样本由M个同步或配准的模态观测及类别标签组成，源域与目标域训练集分别记为：",
        ),
        (
            "Equation",
            "𝒟_d = {((x_(d,i)^(1), …, x_(d,i)^(M)), y_(d,i))}_(i=1)^(N_d),   d ∈ {s,t}.    (1)",
        ),
        (
            "Normal",
            "其中，d=s和d=t分别表示源域与目标域，M=3时对应可见光、红外和AIS。本文采用类别配对的跨域训练设置，即两域训练样本均具有类别标签，并由类平衡采样器在同一批次内构造同类跨域样本对；这种配对只要求类别一致，不要求不同天气下样本一一对应。目标是在保持类别判别性的前提下，学习对天气变化不敏感且不破坏模态互补结构的表示。",
        ),
        (
            "Normal",
            "JMDA-Net的处理链路如图 \\ref{Architecture_diagram} 所示。模态特定编码器首先将三类观测映射到各自的特征空间；高阶模态—域对齐模块利用跨模态张量收缩和域特定投影建立共享表示；双向生成器与两个方向特定判别器继续补偿非线性域差异。由于对抗目标只关心域真实性，网络以类别感知反馈约束映射后的样本语义，并以模态关系漂移损失限制全局域映射对前一阶段模态对齐状态的破坏。分类器在共享表示和源域化目标特征上完成最终预测。",
        ),
        ("Heading 2", "模态特定特征提取"),
        (
            "Normal",
            "可见光、红外和AIS在数据形态、噪声来源及判别线索上差异显著，直接共享底层编码器会迫使网络在低层统计上折中。本文为第m个模态设置独立编码器E_m，并将域d中第i个样本的特征写为：",
        ),
        ("Equation", "h_(d,i)^(m) = E_m(x_(d,i)^(m)) ∈ ℝ^(d_m),   d ∈ {s,t}.    (2)"),
        (
            "Normal",
            "可见光分支采用ResNet-18提取纹理、轮廓和局部结构；红外分支由四级卷积编码器构成，通过逐级下采样和全局池化聚合热轮廓信息。该分支不包含解码器和跳跃连接，因此更准确地说是轻量级层次卷积编码器，而非完整U-Net。AIS分支面向两通道I/Q序列，使用复值一维卷积保持同相分量与正交分量之间的耦合。对复权重W=A+iB和输入X=I+iQ，其卷积写为：",
        ),
        (
            "Equation",
            "W ∗ X = (A ∗ I − B ∗ Q) + i(B ∗ I + A ∗ Q).    (3)",
        ),
        (
            "Normal",
            "各分支最后通过线性层输出统一尺度的高层表示。独立编码避免了早期混合异质统计，而跨模态交互留给后续张量模块完成。",
        ),
        ("Heading 2", "高阶模态—域对齐"),
        (
            "Normal",
            "该模块受OSAN以单阶段方式联合处理模态交互与域对齐的思想启发 \\cite{liu2023osan}，但采用端到端可微的因子化张量收缩，而不是训练过程中交替求解广义特征值问题。对每个模态m，分别设置源域投影U_m∈ℝ^(d_m×r_m)和目标域投影V_m∈ℝ^(d_m×r_m)。两组投影正交初始化，并在每次参数更新后通过QR分解回缩到列正交空间，以降低尺度漂移和数值不稳定。",
        ),
        (
            "Normal",
            "对样本i，将各模态特征的外积记为秩一张量𝒳_(d,i)=h_(d,i)^(1)∘…∘h_(d,i)^(M)。显式构造该张量需要∏_m d_m的存储量。利用秩一外积的可分解性，沿除第n个模态外的其余维度投影并收缩，可等价地写为：",
        ),
        (
            "Equation",
            "a_(d,i)^(n) = ∏_(m≠n) 1^⊤(h_(d,i)^(m) W_(d,m)),   c_(d,i)^(n) = a_(d,i)^(n) h_(d,i)^(n),    (4)",
        ),
        (
            "Normal",
            "其中，W_(s,m)=U_m，W_(t,m)=V_m；标量a_(d,i)^(n)汇集其余模态经投影后的响应，并作为第n个模态的样本自适应门控。该写法与“外积—模乘—收缩”结果等价，但内存复杂度随各模态维度线性增长。由类平衡采样器构造的源—目标同类样本对按批次位置对应，张量对齐损失定义为各模态收缩特征的负平均余弦相关性：",
        ),
        (
            "Equation",
            "𝓛_TAL = −(1/M) ∑_(n=1)^M (1/B) ∑_(i=1)^B 〈c_(s,i)^(n), c_(t,i)^(n)〉/(‖c_(s,i)^(n)‖₂ ‖c_(t,i)^(n)‖₂).    (5)",
        ),
        (
            "Normal",
            "需要指出，式（5）最大化的是配对样本的相关性，并不等同于证明两域总体分布完全重合。优化后，各模态通过对应的域特定投影得到z_(s,i)^(m)=h_(s,i)^(m)U_m和z_(t,i)^(m)=h_(t,i)^(m)V_m，并按固定顺序拼接为f_(s,i)=[z_(s,i)^(1);…;z_(s,i)^(M)]与f_(t,i)=[z_(t,i)^(1);…;z_(t,i)^(M)]。这两个融合向量既保留独立模态块，也作为后续非线性域映射的输入。",
        ),
        ("Heading 2", "双向特征生成与双判别器"),
        (
            "Normal",
            "张量模块通过线性投影约束跨域相关方向，难以覆盖由遮挡、散射和传感器响应变化引起的非线性偏移。为此，本文在融合特征空间中引入源域到目标域映射G_(s→t)和目标域到源域映射G_(t→s)。两个生成器均为带层归一化的残差多层感知机，其输出写为：",
        ),
        (
            "Equation",
            "G_(a→b)(f) = f + ρ tanh(g_(a→b)(f)),   f̂_t = G_(s→t)(f_s),   f̂_s = G_(t→s)(f_t).    (6)",
        ),
        (
            "Normal",
            "残差系数ρ限制单次映射幅度，使生成器优先学习跨域差异而不是重写全部语义。与单一域判别器不同，本文分别设置源域判别器D_s和目标域判别器D_t：D_s区分真实源域特征f_s与由目标域生成的源域化特征f̂_s，D_t区分真实目标域特征f_t与源域生成的目标域化特征f̂_t。令ℓ_bin表示二类交叉熵，判别器目标为：",
        ),
        (
            "Equation",
            "𝓛_D^s = 1/2[ℓ_bin(D_s(f_s),1)+ℓ_bin(D_s(f̂_s),0)],   𝓛_D^t = 1/2[ℓ_bin(D_t(f_t),1)+ℓ_bin(D_t(f̂_t),0)],",
        ),
        (
            "Equation",
            "𝓛_D = 1/2(𝓛_D^s + 𝓛_D^t).    (7)",
        ),
        (
            "Normal",
            "生成器则最小化𝓛_adv=ℓ_bin(D_s(f̂_s),1)+ℓ_bin(D_t(f̂_t),1)，分别使两个方向的生成特征接近其目标分布。方向特定判别避免把“域不可分”简化为单一混合分布，同时也为目标域到源域的推理路径提供直接监督。",
        ),
        ("Heading 2", "类别感知反馈"),
        (
            "Normal",
            "对抗损失只约束生成特征的域归属，不能保证映射前后的样本身份和类别不变。本文从可逆性、最小改写、批内类结构、跨批类别中心和分类决策五个层面提供反馈。首先，双向循环一致性与恒等保持损失分别为：",
        ),
        (
            "Equation",
            "𝓛_cyc = ‖G_(t→s)(G_(s→t)(f_s))−f_s‖₁ + ‖G_(s→t)(G_(t→s)(f_t))−f_t‖₁,    (8)",
        ),
        (
            "Equation",
            "𝓛_id = ‖G_(t→s)(f_s)−f_s‖₁ + ‖G_(s→t)(f_t)−f_t‖₁.    (9)",
        ),
        (
            "Normal",
            "循环一致性约束跨域往返后的可恢复性，恒等保持抑制生成器对已经处于目标分布的输入进行无意义改写。二者只约束特征重建，仍不足以显式拉近同类跨域样本。对生成锚点q_i、另一域真实候选r_j及同类索引集合P(i)={j|y_j=y_i}，类感知批内对比损失写为：",
        ),
        (
            "Equation",
            "𝓛_con(q,r) = −(1/B)∑_i log [∑_(j∈P(i)) exp(sim(q_i,r_j)/τ) / ∑_j exp(sim(q_i,r_j)/τ)],    (10)",
        ),
        (
            "Normal",
            "其中sim为余弦相似度，τ为温度系数。实际训练同时计算𝓛_con(f̂_s,f_s)和𝓛_con(f̂_t,f_t)并取均值；分子聚合批内全部同类样本，避免正样本数量随类别频次变化而改变损失下界。",
        ),
        (
            "Normal",
            "批内对比受小批次类别构成限制。为提供跨批次的稳定参照，本文维护源域和目标域的指数滑动类别原型。对域d、类别c在当前批次的均值p̄_(d,c)，原型更新为p_(d,c)←μp_(d,c)+(1−μ)p̄_(d,c)。生成的源域化特征与源域原型比较，生成的目标域化特征与目标域原型比较，并以真实类别作为交叉熵目标，得到原型对比损失𝓛_proto。不存在于当前批次且尚未初始化的类别原型不参与计算。",
        ),
        (
            "Normal",
            "最后，将生成特征送入共享分类器C，定义𝓛_fb=CE(C(f̂_s),y_t)+CE(C(f̂_t),y_s)。计算该项时固定分类器参数，只允许梯度穿过分类器回传至生成器，因而它约束的是生成特征能否保持原类别，而不会让分类器反向迁就错误映射。分类器本身由真实融合特征监督：𝓛_cls=CE(C(f_s),y_s)+λ_t CE(C(f_t),y_t)。",
        ),
        ("Heading 2", "模态关系漂移约束"),
        (
            "Normal",
            "双向对抗映射作用于完整融合向量，虽然能够缩小全局域差异，却可能同时改变各模态块之间的相对结构，使张量模块已经建立的模态对齐状态退化。漂移模块不承担新的域对齐任务，而是作为后续映射的保护性约束。对融合特征f按投影维度切分得到{z^(m)}_(m=1)^M，并以批内余弦Gram矩阵R^(m)=norm(z^(m))norm(z^(m))^⊤表示第m个模态对样本间关系的刻画。模态关系不一致度定义为：",
        ),
        (
            "Equation",
            "A(f) = [2/(M(M−1))] ∑_(p<q) (1/B²) ‖R^(p)−R^(q)‖_F².    (11)",
        ),
        (
            "Normal",
            "A(f)越小，说明不同模态对批内样本关系的描述越一致。为了允许域映射进行必要调整，本文不要求映射后关系与映射前完全相等，而只惩罚不一致度超过基线与容许边界δ的增量：",
        ),
        (
            "Equation",
            "𝓛_drift = 1/2([A(G_(s→t)(sg(f_s)))−sg(A(f_s))−δ]_+ + [A(G_(t→s)(sg(f_t)))−sg(A(f_t))−δ]_+).    (12)",
        ),
        (
            "Normal",
            "其中sg(·)表示停止梯度，[·]_+=max(·,0)。因此，漂移损失以张量对齐输出为固定参照，只更新双向生成器，而不会推动张量投影去追随生成器产生的变化；当关系变化未超过δ时该项为零。训练初期先关闭该约束，待域映射形成后再线性增加其权重，以避免保护尚未稳定的模态关系。",
        ),
        ("Heading 2", "整体优化与推理"),
        (
            "Normal",
            "训练采用判别器与主网络交替更新。固定编码器、张量投影和生成器时，两个判别器最小化式（7）；固定判别器时，编码器、张量投影、生成器和分类器最小化：",
        ),
        (
            "Equation",
            "𝓛_adv = λ_s𝓛_adv^s + λ_t𝓛_adv^t.    (13)",
        ),
        (
            "Equation",
            "𝓛_C = λ_cyc𝓛_cyc + λ_id𝓛_id + λ_con𝓛_con + λ_proto𝓛_proto + λ_fb𝓛_fb.    (14)",
        ),
        (
            "Equation",
            "𝓛_main = 𝓛_cls + λ_TAL𝓛_TAL + γ_adv𝓛_adv + γ_C𝓛_C + γ_driftλ_drift𝓛_drift.    (15)",
        ),
        (
            "Normal",
            "γ_adv、γ_C和γ_drift分别控制对抗项、类别感知反馈和漂移约束的分阶段启用。该优化顺序使张量模块先建立可用的模态—域共享表示，随后逐步加入非线性域映射及其语义约束，最后再限制模态关系恶化。每次主网络更新后，对U_m和V_m执行QR回缩。推理时仅输入目标域样本，经模态编码和目标域投影得到f_t，再使用G_(t→s)生成源域化特征f̂_s，最终由共享分类器C输出类别预测；判别器、反向生成器和各辅助损失均不参与推理。",
        ),
    ]

    # The source document has no dedicated equation style. Create one from Normal
    # so equations can be formatted independently without changing body text.
    if "Equation" not in [style.name for style in document.styles]:
        document.styles.add_style("Equation", 1)
    for style, text in blocks:
        append_before(document, anchor, text, style)


def main():
    document = Document(SOURCE)
    revise_cross_section_summary(document)
    anchor = remove_method_section(document)
    build_method(document, anchor)
    anchor.paragraph_format.page_break_before = True

    properties = document.core_properties
    properties.title = "用于恶劣天气下异构光学与AIS数据的海上目标识别的双对抗网络（方法修订稿）"
    properties.subject = "方法章节一致性与完整性修订"
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
