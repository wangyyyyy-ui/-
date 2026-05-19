#!/usr/bin/env python3
"""
生成实验设计与评价体系PDF文档
面向课堂复杂场景的多目标视觉感知与学习状态分析研究
"""

import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm, mm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, CondPageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# ============================================================
# 字体注册
# ============================================================
pdfmetrics.registerFont(TTFont('SarasaBold', '/usr/share/fonts/truetype/chinese/SarasaMonoSC-Bold.ttf'))
pdfmetrics.registerFont(TTFont('SarasaRegular', '/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))

registerFontFamily('SarasaRegular', normal='SarasaRegular', bold='SarasaBold')
registerFontFamily('DejaVuSans', normal='DejaVuSans', bold='DejaVuSans')

# ============================================================
# 调色板
# ============================================================
ACCENT       = colors.HexColor('#25738c')
TEXT_PRIMARY  = colors.HexColor('#1b1b19')
TEXT_MUTED    = colors.HexColor('#838077')
BG_SURFACE   = colors.HexColor('#dfddd8')
BG_PAGE      = colors.HexColor('#f2f2f0')

TABLE_HEADER_COLOR = ACCENT
TABLE_HEADER_TEXT  = colors.white
TABLE_ROW_EVEN     = colors.white
TABLE_ROW_ODD      = BG_SURFACE

# ============================================================
# 样式定义
# ============================================================
PAGE_W, PAGE_H = A4
LEFT_MARGIN = 2.2 * cm
RIGHT_MARGIN = 2.2 * cm
TOP_MARGIN = 2.5 * cm
BOTTOM_MARGIN = 2.5 * cm
AVAILABLE_WIDTH = PAGE_W - LEFT_MARGIN - RIGHT_MARGIN

# 标题样式
h1_style = ParagraphStyle(
    name='H1', fontName='SarasaRegular', fontSize=18, leading=28,
    textColor=ACCENT, spaceBefore=18, spaceAfter=12,
    alignment=TA_LEFT, wordWrap='CJK',
)

h2_style = ParagraphStyle(
    name='H2', fontName='SarasaRegular', fontSize=14, leading=22,
    textColor=TEXT_PRIMARY, spaceBefore=14, spaceAfter=8,
    alignment=TA_LEFT, wordWrap='CJK',
)

h3_style = ParagraphStyle(
    name='H3', fontName='SarasaRegular', fontSize=12, leading=20,
    textColor=TEXT_PRIMARY, spaceBefore=10, spaceAfter=6,
    alignment=TA_LEFT, wordWrap='CJK',
)

# 正文样式
body_style = ParagraphStyle(
    name='Body', fontName='SarasaRegular', fontSize=10.5, leading=18,
    textColor=TEXT_PRIMARY, spaceBefore=0, spaceAfter=6,
    alignment=TA_LEFT, firstLineIndent=21, wordWrap='CJK',
)

body_no_indent = ParagraphStyle(
    name='BodyNoIndent', fontName='SarasaRegular', fontSize=10.5, leading=18,
    textColor=TEXT_PRIMARY, spaceBefore=0, spaceAfter=6,
    alignment=TA_LEFT, wordWrap='CJK',
)

# 表格样式
table_header_style = ParagraphStyle(
    name='TableHeader', fontName='SarasaRegular', fontSize=10, leading=15,
    textColor=TABLE_HEADER_TEXT, alignment=TA_CENTER, wordWrap='CJK',
)

table_cell_style = ParagraphStyle(
    name='TableCell', fontName='SarasaRegular', fontSize=9.5, leading=14,
    textColor=TEXT_PRIMARY, alignment=TA_CENTER, wordWrap='CJK',
)

table_cell_left = ParagraphStyle(
    name='TableCellLeft', fontName='SarasaRegular', fontSize=9.5, leading=14,
    textColor=TEXT_PRIMARY, alignment=TA_LEFT, wordWrap='CJK',
)

# 公式样式
formula_style = ParagraphStyle(
    name='Formula', fontName='DejaVuSans', fontSize=10, leading=16,
    textColor=TEXT_PRIMARY, spaceBefore=6, spaceAfter=6,
    alignment=TA_CENTER,
)

# 列表样式
list_style = ParagraphStyle(
    name='ListItem', fontName='SarasaRegular', fontSize=10.5, leading=18,
    textColor=TEXT_PRIMARY, spaceBefore=2, spaceAfter=2,
    leftIndent=24, firstLineIndent=-12, wordWrap='CJK',
)

# ============================================================
# 辅助函数
# ============================================================
def make_table(data, col_widths=None, has_header=True):
    """创建标准格式表格"""
    if col_widths is None:
        col_widths = [AVAILABLE_WIDTH / len(data[0])] * len(data[0])
    
    t = Table(data, colWidths=col_widths, hAlign='CENTER')
    style_cmds = [
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, TEXT_MUTED),
    ]
    if has_header:
        style_cmds.extend([
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), TABLE_HEADER_TEXT),
        ])
        for i in range(1, len(data)):
            bg = TABLE_ROW_EVEN if i % 2 == 1 else TABLE_ROW_ODD
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), bg))
    
    t.setStyle(TableStyle(style_cmds))
    return t


def h1(text):
    return Paragraph(f'<b>{text}</b>', h1_style)

def h2(text):
    return Paragraph(f'<b>{text}</b>', h2_style)

def h3(text):
    return Paragraph(f'<b>{text}</b>', h3_style)

def body(text):
    return Paragraph(text, body_style)

def body_ni(text):
    return Paragraph(text, body_no_indent)

def li(text):
    return Paragraph(f'- {text}', list_style)

def formula(text):
    return Paragraph(text, formula_style)


# ============================================================
# 文档内容
# ============================================================
def build_document():
    output_path = '/home/z/my-project/download/classroom-vision/docs/实验设计与评价体系.pdf'
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title='面向课堂复杂场景的多目标视觉感知与学习状态分析 - 实验设计与评价体系',
        author='110实验室前沿技术分组',
    )
    
    story = []
    
    # ============================================================
    # 封面页
    # ============================================================
    story.append(Spacer(1, 80))
    story.append(Paragraph(
        '<b>面向课堂复杂场景的</b>',
        ParagraphStyle('CoverTitle1', fontName='SarasaRegular', fontSize=24,
                       leading=36, textColor=ACCENT, alignment=TA_CENTER, wordWrap='CJK')
    ))
    story.append(Paragraph(
        '<b>多目标视觉感知与学习状态分析研究</b>',
        ParagraphStyle('CoverTitle2', fontName='SarasaRegular', fontSize=24,
                       leading=36, textColor=ACCENT, alignment=TA_CENTER, wordWrap='CJK')
    ))
    story.append(Spacer(1, 30))
    story.append(Paragraph(
        '实验设计与评价体系文档',
        ParagraphStyle('CoverSub', fontName='SarasaRegular', fontSize=16,
                       leading=24, textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')
    ))
    story.append(Spacer(1, 60))
    
    cover_info = [
        ['项目名称', '面向课堂复杂场景的多目标视觉感知与学习状态分析研究'],
        ['指导教师', '周灿宇'],
        ['带教学长', '李腊全'],
        ['研究团队', '110实验室前沿技术分组'],
        ['文档版本', 'V1.0'],
        ['文档日期', '2026年5月'],
    ]
    cover_table = Table(cover_info, colWidths=[AVAILABLE_WIDTH * 0.25, AVAILABLE_WIDTH * 0.65], hAlign='CENTER')
    cover_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'SarasaRegular'),
        ('FONTNAME', (1, 0), (1, -1), 'SarasaRegular'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('LEADING', (0, 0), (-1, -1), 18),
        ('TEXTCOLOR', (0, 0), (0, -1), ACCENT),
        ('TEXTCOLOR', (1, 0), (1, -1), TEXT_PRIMARY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, BG_SURFACE),
    ]))
    story.append(cover_table)
    
    story.append(PageBreak())
    
    # ============================================================
    # 目录
    # ============================================================
    story.append(Paragraph('<b>目  录</b>', ParagraphStyle(
        'TOCTitle', fontName='SarasaRegular', fontSize=18, leading=28,
        textColor=ACCENT, alignment=TA_CENTER, spaceAfter=24, wordWrap='CJK'
    )))
    
    toc_items = [
        ('1', '实验总体设计'),
        ('  1.1', '实验目标'),
        ('  1.2', '实验环境'),
        ('  1.3', '数据集规划'),
        ('2', '实验方案设计'),
        ('  2.1', '实验一：目标检测改进实验'),
        ('  2.2', '实验二：多目标跟踪实验'),
        ('  2.3', '实验三：人脸识别增强实验'),
        ('  2.4', '实验四：课堂状态分析实验'),
        ('  2.5', '实验五：端到端系统集成实验'),
        ('3', '评价指标体系'),
        ('  3.1', '目标检测评价指标'),
        ('  3.2', '多目标跟踪评价指标'),
        ('  3.3', '人脸识别评价指标'),
        ('  3.4', '课堂状态分析评价指标'),
        ('  3.5', '系统性能评价指标'),
        ('4', '评价指标计算方法'),
        ('  4.1', 'Precision与Recall'),
        ('  4.2', 'mAP计算方法'),
        ('  4.3', 'MOTA计算方法'),
        ('  4.4', 'EER计算方法'),
        ('  4.5', 'Cohen\'s Kappa'),
        ('5', '实验结果记录模板'),
        ('6', '评估工具使用说明'),
    ]
    
    toc_style_1 = ParagraphStyle('TOC1', fontName='SarasaRegular', fontSize=11, leading=22,
                                  textColor=TEXT_PRIMARY, wordWrap='CJK')
    toc_style_2 = ParagraphStyle('TOC2', fontName='SarasaRegular', fontSize=10, leading=20,
                                  textColor=TEXT_MUTED, leftIndent=20, wordWrap='CJK')
    
    for num, title in toc_items:
        style = toc_style_1 if not num.startswith(' ') else toc_style_2
        story.append(Paragraph(f'{num}  {title}', style))
    
    story.append(PageBreak())
    
    # ============================================================
    # 第1章：实验总体设计
    # ============================================================
    story.append(h1('1  实验总体设计'))
    story.append(Spacer(1, 12))
    
    story.append(h2('1.1  实验目标'))
    story.append(body(
        '本研究的实验设计围绕以下核心目标展开：验证目标检测改进方案的有效性，评估YOLOv10在课堂场景下的检测性能，'
        '验证Mosaic-9数据增强、BiFPN多尺度特征融合、SE通道注意力机制等改进策略对检测精度的提升效果；'
        '验证ByteTrack在课堂密集遮挡场景的跟踪能力，评估其两轮关联策略在学生密集排列、频繁遮挡条件下的跟踪稳定性，'
        '探索运动模型与弱特征关联的改进方向；验证ArcFace在远距离/遮挡场景的识别鲁棒性，评估ArcFace特征提取在课堂远距离拍摄条件下的人脸识别性能，'
        '探索跟踪辅助和SAM分割辅助两种增强策略；验证多模态融合状态分类的准确性，评估头部姿态、视线方向、身体姿态等多模态信息融合对学生课堂状态分类的效果，'
        '验证时序融合策略对分类稳定性的改善；验证端到端系统的实用性，评估完整流水线在真实课堂场景下的综合性能，包括处理速度、统计准确度和实际可用性。'
    ))
    
    story.append(h2('1.2  实验环境'))
    env_data = [
        [Paragraph('<b>项目</b>', table_header_style), Paragraph('<b>最低要求</b>', table_header_style), Paragraph('<b>推荐配置</b>', table_header_style)],
        [Paragraph('操作系统', table_cell_style), Paragraph('Ubuntu 18.04 / Windows 10', table_cell_style), Paragraph('Ubuntu 20.04+', table_cell_style)],
        [Paragraph('GPU', table_cell_style), Paragraph('NVIDIA GTX 1060 (6GB)', table_cell_style), Paragraph('NVIDIA RTX 3060 (12GB)+', table_cell_style)],
        [Paragraph('CPU', table_cell_style), Paragraph('Intel i5-10400', table_cell_style), Paragraph('Intel i7-12700', table_cell_style)],
        [Paragraph('内存', table_cell_style), Paragraph('16 GB DDR4', table_cell_style), Paragraph('32 GB DDR4', table_cell_style)],
        [Paragraph('Python', table_cell_style), Paragraph('3.8', table_cell_style), Paragraph('3.10+', table_cell_style)],
        [Paragraph('PyTorch', table_cell_style), Paragraph('2.0+', table_cell_style), Paragraph('2.1+', table_cell_style)],
        [Paragraph('CUDA', table_cell_style), Paragraph('11.0', table_cell_style), Paragraph('11.8+', table_cell_style)],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(env_data, [AVAILABLE_WIDTH * 0.2, AVAILABLE_WIDTH * 0.38, AVAILABLE_WIDTH * 0.38]))
    story.append(Spacer(1, 12))
    
    story.append(h2('1.3  数据集规划'))
    story.append(body(
        '数据集的构建是实验设计的基础环节。本研究采用公开数据集与自建数据集相结合的策略，确保实验结果的可靠性和泛化能力。'
        '目标检测模块以COCO 2017数据集的person类别为基础（约41,000张训练图像，5,000张验证图像），并自建课堂场景数据集，'
        '涵盖小教室场景（20-30人，近距离拍摄，光照良好）、大教室场景（50-80人，远距离拍摄，存在遮挡）、低光照场景以及在线课堂场景。'
        '标注格式采用YOLO格式（class_id cx cy w h），并应用Mosaic-9、MixUp、随机仿射变换、HSV颜色空间增强等数据增强策略。'
    ))
    story.append(body(
        '多目标跟踪模块以MOT17公开数据集（7个训练序列+7个测试序列）为基础，并自建课堂跟踪数据集，从课堂视频中提取连续帧并标注跟踪ID，'
        '标注格式采用MOT格式（frame_id, track_id, x, y, w, h, conf, class_id）。'
        '人脸识别模块的注册集采用学生登记照片（每人1-3张正面照），测试集从课堂视频中截取人脸区域，'
        '包含正面人脸（近距离）、侧面人脸（偏转30-60度）、遮挡人脸（口罩、手、书本遮挡）和远距离人脸（模糊、低分辨率）等不同难度场景。'
        '课堂状态数据集采用人工逐帧标注，状态类别包括专注、分心、低头、离座、举手五类，标注频率为每秒1次，'
        '要求两名标注员独立标注，不一致样本由第三名标注员裁决，标注员间一致性Kappa系数需大于0.8。'
    ))
    
    # ============================================================
    # 第2章：实验方案设计
    # ============================================================
    story.append(Spacer(1, 18))
    story.append(h1('2  实验方案设计'))
    story.append(Spacer(1, 12))
    
    # 实验一
    story.append(h2('2.1  实验一：目标检测改进实验'))
    story.append(body(
        '本实验旨在验证YOLOv10在课堂场景下的检测性能，并评估Mosaic-9数据增强、BiFPN多尺度特征融合、SE通道注意力机制等改进策略的有效性。'
        '实验采用控制变量法，逐步添加改进策略，对比各策略对检测精度和推理速度的影响。'
    ))
    
    det_exp_data = [
        [Paragraph('<b>实验编号</b>', table_header_style), Paragraph('<b>实验内容</b>', table_header_style), Paragraph('<b>对比模型/方法</b>', table_header_style), Paragraph('<b>评估指标</b>', table_header_style)],
        [Paragraph('Exp1-1', table_cell_style), Paragraph('YOLOv10基线实验', table_cell_left), Paragraph('yolov10n/s/m/l/x', table_cell_left), Paragraph('P, R, mAP50, mAP50-95, FPS', table_cell_left)],
        [Paragraph('Exp1-2', table_cell_style), Paragraph('Mosaic-9增强', table_cell_left), Paragraph('Mosaic-4 vs Mosaic-9', table_cell_left), Paragraph('mAP50, mAP50-95', table_cell_left)],
        [Paragraph('Exp1-3', table_cell_style), Paragraph('BiFPN融合', table_cell_left), Paragraph('FPN vs BiFPN', table_cell_left), Paragraph('mAP50, mAP50-95, FPS', table_cell_left)],
        [Paragraph('Exp1-4', table_cell_style), Paragraph('SE注意力', table_cell_left), Paragraph('无SE vs 有SE', table_cell_left), Paragraph('mAP50, mAP50-95, FPS', table_cell_left)],
        [Paragraph('Exp1-5', table_cell_style), Paragraph('RT-DETR对比', table_cell_left), Paragraph('YOLOv10 vs RT-DETR', table_cell_left), Paragraph('mAP50, mAP50-95, FPS', table_cell_left)],
        [Paragraph('Exp1-6', table_cell_style), Paragraph('综合改进', table_cell_left), Paragraph('基线 vs 综合改进', table_cell_left), Paragraph('P, R, mAP50, mAP50-95, FPS', table_cell_left)],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(det_exp_data, [AVAILABLE_WIDTH * 0.12, AVAILABLE_WIDTH * 0.22, AVAILABLE_WIDTH * 0.30, AVAILABLE_WIDTH * 0.32]))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表1  目标检测改进实验设计', ParagraphStyle('Caption', fontName='SarasaRegular', fontSize=9, leading=14, textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')))
    story.append(Spacer(1, 12))
    
    story.append(body(
        '训练参数设置如下：输入分辨率640x640，批大小16（受GPU显存限制可调整），训练轮数300（含早停机制，耐心值50），'
        '优化器采用SGD（初始学习率0.01，动量0.937，权重衰减0.0005），学习率调度采用余弦退火策略。'
        '数据增强策略包括Mosaic-9（训练前90%轮次启用）、MixUp（概率0.1）、随机仿射变换（平移0.1，缩放0.5-1.5，旋转0-10度）、'
        'HSV颜色空间增强（色调0.015，饱和度0.7，明度0.4）、随机水平翻转（概率0.5）。'
    ))
    
    # 实验二
    story.append(h2('2.2  实验二：多目标跟踪实验'))
    story.append(body(
        '本实验旨在评估ByteTrack在课堂密集遮挡场景下的跟踪性能，并探索运动模型改进与弱特征关联增强两种改进方向。'
        'ByteTrack的核心思想是同时利用高置信度和低置信度检测结果进行关联：高置信度检测框构建可靠轨迹用于第一轮匹配，'
        '低置信度检测框在第二轮匹配中关联遮挡或模糊目标。这种策略有效减少了目标丢失和身份切换（ID Switch）问题，'
        '特别适合课堂密集场景中存在遮挡的情况。'
    ))
    
    track_exp_data = [
        [Paragraph('<b>实验编号</b>', table_header_style), Paragraph('<b>实验内容</b>', table_header_style), Paragraph('<b>对比方法</b>', table_header_style), Paragraph('<b>评估指标</b>', table_header_style)],
        [Paragraph('Exp2-1', table_cell_style), Paragraph('ByteTrack基线', table_cell_left), Paragraph('DeepSORT, BoT-SORT', table_cell_left), Paragraph('MOTA, IDF1, ID Switch, MT/ML', table_cell_left)],
        [Paragraph('Exp2-2', table_cell_style), Paragraph('运动模型改进', table_cell_left), Paragraph('卡尔曼滤波 vs 改进运动模型', table_cell_left), Paragraph('MOTA, ID Switch', table_cell_left)],
        [Paragraph('Exp2-3', table_cell_style), Paragraph('弱特征关联', table_cell_left), Paragraph('IoU匹配 vs 外观+IoU', table_cell_left), Paragraph('MOTA, IDF1', table_cell_left)],
        [Paragraph('Exp2-4', table_cell_style), Paragraph('综合改进', table_cell_left), Paragraph('基线 vs 综合改进', table_cell_left), Paragraph('MOTA, IDF1, ID Switch, FPS', table_cell_left)],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(track_exp_data, [AVAILABLE_WIDTH * 0.12, AVAILABLE_WIDTH * 0.22, AVAILABLE_WIDTH * 0.32, AVAILABLE_WIDTH * 0.30]))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表2  多目标跟踪实验设计', ParagraphStyle('Caption', fontName='SarasaRegular', fontSize=9, leading=14, textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')))
    story.append(Spacer(1, 12))
    
    # 实验三
    story.append(h2('2.3  实验三：人脸识别增强实验'))
    story.append(body(
        '本实验旨在评估ArcFace在课堂远距离/遮挡场景下的识别鲁棒性，并验证跟踪辅助和SAM分割辅助两种增强策略的有效性。'
        '在课堂场景中，远距离拍摄导致人脸分辨率低、特征模糊，而遮挡（口罩、书本、手）进一步增加了识别难度。'
        '本研究提出两种增强策略：跟踪辅助策略利用多目标跟踪的时间连续性，当某帧人脸识别失败时，利用前后帧的识别结果进行身份传递，'
        '有效解决单帧识别困难的问题；SAM分割辅助策略利用Segment Anything Model（SAM）精确分割人脸区域，'
        '去除背景干扰后再进行特征提取，提高遮挡场景下的识别准确率。'
    ))
    
    face_exp_data = [
        [Paragraph('<b>实验编号</b>', table_header_style), Paragraph('<b>实验内容</b>', table_header_style), Paragraph('<b>对比方法</b>', table_header_style), Paragraph('<b>评估指标</b>', table_header_style)],
        [Paragraph('Exp3-1', table_cell_style), Paragraph('ArcFace基线', table_cell_left), Paragraph('FaceNet, CosFace', table_cell_left), Paragraph('Accuracy, FAR, FRR, EER', table_cell_left)],
        [Paragraph('Exp3-2', table_cell_style), Paragraph('跟踪辅助增强', table_cell_left), Paragraph('单帧识别 vs 跟踪辅助', table_cell_left), Paragraph('Accuracy, FRR, ID Switch', table_cell_left)],
        [Paragraph('Exp3-3', table_cell_style), Paragraph('SAM分割辅助', table_cell_left), Paragraph('原始裁剪 vs SAM分割', table_cell_left), Paragraph('Accuracy, FAR, FRR', table_cell_left)],
        [Paragraph('Exp3-4', table_cell_style), Paragraph('综合增强', table_cell_left), Paragraph('基线 vs 综合增强', table_cell_left), Paragraph('Accuracy, FAR, FRR, EER', table_cell_left)],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(face_exp_data, [AVAILABLE_WIDTH * 0.12, AVAILABLE_WIDTH * 0.22, AVAILABLE_WIDTH * 0.32, AVAILABLE_WIDTH * 0.30]))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表3  人脸识别增强实验设计', ParagraphStyle('Caption', fontName='SarasaRegular', fontSize=9, leading=14, textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')))
    story.append(Spacer(1, 12))
    
    # 实验四
    story.append(h2('2.4  实验四：课堂状态分析实验'))
    story.append(body(
        '本实验旨在评估多模态信息融合对学生课堂状态分类的效果，并验证时序融合策略对分类稳定性的改善。'
        '课堂状态分类是本系统的核心输出之一，直接影响教学辅助统计数据的准确性。'
        '本研究采用分层规则引擎与可选轻量级分类器的混合方案：规则引擎基于头部姿态和身体姿态的阈值判断，'
        '优先级从高到低依次为离座检测（跟踪丢失）、举手检测（身体姿态）、低头检测（头部姿态）、分心检测（视线方向）、专注状态（默认）。'
        '时序融合采用滑动窗口投票与状态机平滑相结合的策略，消除短时抖动，输出稳定的课堂状态统计。'
    ))
    
    state_exp_data = [
        [Paragraph('<b>实验编号</b>', table_header_style), Paragraph('<b>实验内容</b>', table_header_style), Paragraph('<b>对比方法</b>', table_header_style), Paragraph('<b>评估指标</b>', table_header_style)],
        [Paragraph('Exp4-1', table_cell_style), Paragraph('头部姿态估计', table_cell_left), Paragraph('MediaPipe vs OpenFace', table_cell_left), Paragraph('角度误差, FPS', table_cell_left)],
        [Paragraph('Exp4-2', table_cell_style), Paragraph('规则引擎分类', table_cell_left), Paragraph('不同阈值组合', table_cell_left), Paragraph('Accuracy, F1', table_cell_left)],
        [Paragraph('Exp4-3', table_cell_style), Paragraph('多模态融合', table_cell_left), Paragraph('单模态 vs 多模态', table_cell_left), Paragraph('Accuracy, F1, Kappa', table_cell_left)],
        [Paragraph('Exp4-4', table_cell_style), Paragraph('时序融合', table_cell_left), Paragraph('无时序 vs 滑动窗口 vs 状态机', table_cell_left), Paragraph('Accuracy, F1, 稳定性', table_cell_left)],
        [Paragraph('Exp4-5', table_cell_style), Paragraph('综合状态分析', table_cell_left), Paragraph('基线 vs 综合方案', table_cell_left), Paragraph('Accuracy, F1, Kappa', table_cell_left)],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(state_exp_data, [AVAILABLE_WIDTH * 0.12, AVAILABLE_WIDTH * 0.22, AVAILABLE_WIDTH * 0.32, AVAILABLE_WIDTH * 0.30]))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表4  课堂状态分析实验设计', ParagraphStyle('Caption', fontName='SarasaRegular', fontSize=9, leading=14, textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')))
    story.append(Spacer(1, 12))
    
    # 实验五
    story.append(h2('2.5  实验五：端到端系统集成实验'))
    story.append(body(
        '本实验旨在评估完整流水线在真实课堂场景下的综合性能，包括处理速度、统计准确度和实际可用性。'
        '端到端系统将所有子模块串联为完整的课堂分析流水线：视频输入、预处理、目标检测、多目标跟踪、人脸识别、头部姿态估计、'
        '状态分类、时序融合、可视化输出和统计导出。系统支持实时模式（处理摄像头或RTSP流）和离线模式（处理视频文件）两种运行方式。'
        '评估维度包括端到端延迟（各模块延迟分解）、吞吐量（FPS）、统计准确度（到课率偏差、抬头率偏差、离座次数偏差）和GPU显存占用。'
    ))
    
    # ============================================================
    # 第3章：评价指标体系
    # ============================================================
    story.append(Spacer(1, 18))
    story.append(h1('3  评价指标体系'))
    story.append(Spacer(1, 12))
    
    story.append(body(
        '本研究建立了覆盖四个核心模块的完整评价指标体系，从检测精度、跟踪稳定性、识别鲁棒性和状态分类准确性四个维度全面评估系统性能。'
        '各模块评价指标的定义、计算方法和课堂场景意义详述如下。'
    ))
    
    # 3.1 目标检测评价指标
    story.append(h2('3.1  目标检测评价指标'))
    
    det_metrics_data = [
        [Paragraph('<b>指标名称</b>', table_header_style), Paragraph('<b>英文缩写</b>', table_header_style), Paragraph('<b>定义</b>', table_header_style), Paragraph('<b>课堂场景意义</b>', table_header_style)],
        [Paragraph('精确率', table_cell_style), Paragraph('Precision', table_cell_style), Paragraph('TP / (TP + FP)', table_cell_style), Paragraph('检测结果的"可信度"', table_cell_left)],
        [Paragraph('召回率', table_cell_style), Paragraph('Recall', table_cell_style), Paragraph('TP / (TP + FN)', table_cell_style), Paragraph('检测器的"完整性"', table_cell_left)],
        [Paragraph('平均精度', table_cell_style), Paragraph('mAP@0.5', table_cell_style), Paragraph('IoU=0.5时各类AP均值', table_cell_style), Paragraph('检测定位学生的整体能力', table_cell_left)],
        [Paragraph('严格平均精度', table_cell_style), Paragraph('mAP@0.5:0.95', table_cell_style), Paragraph('多IoU阈值下mAP均值', table_cell_style), Paragraph('检测框定位精确度', table_cell_left)],
        [Paragraph('推理速度', table_cell_style), Paragraph('FPS', table_cell_style), Paragraph('每秒处理帧数', table_cell_style), Paragraph('实时性评估', table_cell_left)],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(det_metrics_data, [AVAILABLE_WIDTH * 0.15, AVAILABLE_WIDTH * 0.18, AVAILABLE_WIDTH * 0.30, AVAILABLE_WIDTH * 0.33]))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表5  目标检测评价指标', ParagraphStyle('Caption', fontName='SarasaRegular', fontSize=9, leading=14, textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')))
    story.append(Spacer(1, 12))
    
    # 3.2 多目标跟踪评价指标
    story.append(h2('3.2  多目标跟踪评价指标'))
    
    track_metrics_data = [
        [Paragraph('<b>指标名称</b>', table_header_style), Paragraph('<b>英文缩写</b>', table_header_style), Paragraph('<b>定义</b>', table_header_style), Paragraph('<b>课堂场景意义</b>', table_header_style)],
        [Paragraph('多目标跟踪精度', table_cell_style), Paragraph('MOTA', table_cell_style), Paragraph('1 - (FN + FP + IDSW) / GT', table_cell_style), Paragraph('跟踪整体质量', table_cell_left)],
        [Paragraph('ID F1分数', table_cell_style), Paragraph('IDF1', table_cell_style), Paragraph('身份识别的F1分数', table_cell_style), Paragraph('身份保持能力', table_cell_left)],
        [Paragraph('身份切换次数', table_cell_style), Paragraph('ID Switch', table_cell_style), Paragraph('同一目标ID变化次数', table_cell_style), Paragraph('跟踪稳定性', table_cell_left)],
        [Paragraph('主要跟踪目标数', table_cell_style), Paragraph('MT', table_cell_style), Paragraph('跟踪覆盖率>=80%的目标数', table_cell_style), Paragraph('长期跟踪能力', table_cell_left)],
        [Paragraph('主要丢失目标数', table_cell_style), Paragraph('ML', table_cell_style), Paragraph('跟踪覆盖率<=20%的目标数', table_cell_style), Paragraph('跟踪丢失程度', table_cell_left)],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(track_metrics_data, [AVAILABLE_WIDTH * 0.17, AVAILABLE_WIDTH * 0.15, AVAILABLE_WIDTH * 0.32, AVAILABLE_WIDTH * 0.32]))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表6  多目标跟踪评价指标', ParagraphStyle('Caption', fontName='SarasaRegular', fontSize=9, leading=14, textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')))
    story.append(Spacer(1, 12))
    
    # 3.3 人脸识别评价指标
    story.append(h2('3.3  人脸识别评价指标'))
    
    face_metrics_data = [
        [Paragraph('<b>指标名称</b>', table_header_style), Paragraph('<b>英文缩写</b>', table_header_style), Paragraph('<b>定义</b>', table_header_style), Paragraph('<b>课堂场景意义</b>', table_header_style)],
        [Paragraph('识别准确率', table_cell_style), Paragraph('Accuracy', table_cell_style), Paragraph('正确识别数 / 总识别数', table_cell_style), Paragraph('整体识别性能', table_cell_left)],
        [Paragraph('误签率', table_cell_style), Paragraph('FAR', table_cell_style), Paragraph('冒充者被接受的比例', table_cell_style), Paragraph('安全性评估', table_cell_left)],
        [Paragraph('漏签率', table_cell_style), Paragraph('FRR', table_cell_style), Paragraph('真实用户被拒绝的比例', table_cell_style), Paragraph('便利性评估', table_cell_left)],
        [Paragraph('等错误率', table_cell_style), Paragraph('EER', table_cell_style), Paragraph('FAR = FRR时的错误率', table_cell_style), Paragraph('系统整体性能', table_cell_left)],
        [Paragraph('首位命中率', table_cell_style), Paragraph('Rank-1', table_cell_style), Paragraph('正确身份排在第一位的比例', table_cell_style), Paragraph('识别排序质量', table_cell_left)],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(face_metrics_data, [AVAILABLE_WIDTH * 0.15, AVAILABLE_WIDTH * 0.15, AVAILABLE_WIDTH * 0.32, AVAILABLE_WIDTH * 0.33]))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表7  人脸识别评价指标', ParagraphStyle('Caption', fontName='SarasaRegular', fontSize=9, leading=14, textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')))
    story.append(Spacer(1, 12))
    
    # 3.4 课堂状态分析评价指标
    story.append(h2('3.4  课堂状态分析评价指标'))
    
    state_metrics_data = [
        [Paragraph('<b>指标名称</b>', table_header_style), Paragraph('<b>英文缩写</b>', table_header_style), Paragraph('<b>定义</b>', table_header_style), Paragraph('<b>课堂场景意义</b>', table_header_style)],
        [Paragraph('总体分类准确率', table_cell_style), Paragraph('Accuracy', table_cell_style), Paragraph('正确分类数 / 总样本数', table_cell_style), Paragraph('状态分类整体性能', table_cell_left)],
        [Paragraph('各类别F1分数', table_cell_style), Paragraph('F1-Score', table_cell_style), Paragraph('2PR / (P + R)', table_cell_style), Paragraph('各类别分类平衡性', table_cell_left)],
        [Paragraph('一致性系数', table_cell_style), Paragraph("Cohen's Kappa", table_cell_style), Paragraph('校正随机一致后的一致性', table_cell_style), Paragraph('分类可靠性', table_cell_left)],
        [Paragraph('混淆矩阵', table_cell_style), Paragraph('CM', table_cell_style), Paragraph('预测vs真实标签矩阵', table_cell_style), Paragraph('错误模式分析', table_cell_left)],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(state_metrics_data, [AVAILABLE_WIDTH * 0.17, AVAILABLE_WIDTH * 0.18, AVAILABLE_WIDTH * 0.30, AVAILABLE_WIDTH * 0.30]))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表8  课堂状态分析评价指标', ParagraphStyle('Caption', fontName='SarasaRegular', fontSize=9, leading=14, textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')))
    story.append(Spacer(1, 12))
    
    # 3.5 系统性能评价指标
    story.append(h2('3.5  系统性能评价指标'))
    story.append(body(
        '系统性能评价指标从端到端角度评估完整流水线的运行效率和资源消耗，包括端到端延迟（各模块延迟分解，目标检测约5-15ms、'
        '多目标跟踪约2-5ms、人脸识别约10-20ms、姿态估计约5-10ms、状态分析约1ms、可视化约3ms）、吞吐量（FPS，实时模式要求>=15FPS，'
        '离线模式要求>=30FPS）、GPU显存占用（峰值显存不超过GPU容量的80%）以及统计准确度（到课率偏差在正负5%以内，'
        '抬头率偏差在正负10%以内，离座次数偏差在正负20%以内）。'
    ))
    
    # ============================================================
    # 第4章：评价指标计算方法
    # ============================================================
    story.append(Spacer(1, 18))
    story.append(h1('4  评价指标计算方法'))
    story.append(Spacer(1, 12))
    
    story.append(h2('4.1  Precision与Recall'))
    story.append(body(
        '精确率（Precision）定义为在所有被模型预测为正例的样本中，实际为正例的比例，计算公式为 Precision = TP / (TP + FP)，'
        '其中TP为正确检测到的目标数，FP为误检数。召回率（Recall）定义为在所有实际为正例的样本中，被正确检测为正例的比例，'
        '计算公式为 Recall = TP / (TP + FN)，其中FN为漏检数。在课堂场景中，高Precision意味着系统报告的检测结果大部分是真实的，'
        '减少误报对后续跟踪和识别的干扰；高Recall意味着系统能检测到画面中的大部分学生，减少漏检对到课率统计的影响。'
    ))
    
    story.append(h2('4.2  mAP计算方法'))
    story.append(body(
        'Average Precision（AP）是Precision-Recall曲线下的面积，综合反映不同置信度阈值下的检测性能。'
        '计算方法采用11点插值法：AP = (1/11) x max(Precision | Recall >= t)，其中t取值0, 0.1, 0.2, ..., 1.0。'
        'mAP@0.5是IoU阈值为0.5时所有类别AP的平均值，其中IoU（Intersection over Union）定义为 '
        'IoU = Area(B_pred 与 B_gt) / Area(B_pred 或 B_gt)，当IoU >= 0.5时认为检测框与真值框匹配成功。'
        'mAP@0.5:0.95是在IoU阈值从0.5到0.95（步长0.05）的10个阈值下mAP的平均值，是更严格的评估指标，'
        '要求检测框与真值框高度重合，有利于后续人脸识别和姿态估计的精度。'
    ))
    
    story.append(h2('4.3  MOTA计算方法'))
    story.append(body(
        'MOTA（Multi-Object Tracking Accuracy）是多目标跟踪最核心的评价指标，综合考虑了漏检、误检和身份切换三种错误，'
        '计算公式为 MOTA = 1 - (FN + FP + IDSW) / GT，其中FN为总漏检数，FP为总误检数，IDSW为身份切换总数，'
        'GT为真值目标总数。MOTA的取值范围为(-inf, 1]，值越大表示跟踪性能越好。'
        '当误检和身份切换过多时，MOTA可能为负值。在课堂场景中，MOTA反映跟踪器在密集遮挡条件下维持稳定跟踪的综合能力。'
    ))
    
    story.append(h2("4.4  EER计算方法"))
    story.append(body(
        'EER（Equal Error Rate）是FAR等于FRR时的错误率，是生物特征识别系统的标准评估指标。'
        '计算方法为：收集真实匹配的相似度分数（genuine scores）和冒充匹配的相似度分数（impostor scores），'
        '绘制ROC曲线（FPR vs TPR），找到FNR = FPR的点，该点对应的错误率即为EER。'
        'EER越低，系统性能越好。在课堂人脸识别场景中，EER提供了一个与阈值选择无关的系统性能度量，'
        '便于不同系统之间的公平比较。'
    ))
    
    story.append(h2("4.5  Cohen's Kappa"))
    story.append(body(
        "Cohen's Kappa系数是衡量分类一致性的统计量，校正了随机一致性的影响，计算公式为 "
        "Kappa = (Po - Pe) / (1 - Pe)，其中Po为观察一致率（即总体分类准确率），"
        "Pe为随机一致率（由各类别边际分布计算得到）。Kappa的取值范围为[-1, 1]，"
        "通常认为Kappa > 0.8表示一致性很好，0.6-0.8表示一致性较好，0.4-0.6表示一致性中等，"
        "< 0.4表示一致性较差。在课堂状态分析中，Cohen's Kappa用于评估分类结果与人工标注之间的一致性，"
        "以及不同分类方法之间的一致性，是比单纯准确率更可靠的评估指标。"
    ))
    
    # ============================================================
    # 第5章：实验结果记录模板
    # ============================================================
    story.append(Spacer(1, 18))
    story.append(h1('5  实验结果记录模板'))
    story.append(Spacer(1, 12))
    
    story.append(body(
        '为确保实验结果的可追溯性和可复现性，本研究为每类实验设计了标准化的结果记录模板。'
        '所有实验结果以JSON格式存储，便于自动化分析和对比。以下为各类实验的记录模板示例。'
    ))
    
    story.append(h3('5.1  检测实验记录'))
    det_template = [
        [Paragraph('<b>字段</b>', table_header_style), Paragraph('<b>说明</b>', table_header_style), Paragraph('<b>示例值</b>', table_header_style)],
        [Paragraph('experiment_id', table_cell_style), Paragraph('实验编号', table_cell_left), Paragraph('Exp1-2', table_cell_style)],
        [Paragraph('model', table_cell_style), Paragraph('模型名称', table_cell_left), Paragraph('yolov10s', table_cell_style)],
        [Paragraph('improvements', table_cell_style), Paragraph('改进策略列表', table_cell_left), Paragraph('["Mosaic-9"]', table_cell_style)],
        [Paragraph('precision', table_cell_style), Paragraph('精确率', table_cell_left), Paragraph('0.892', table_cell_style)],
        [Paragraph('recall', table_cell_style), Paragraph('召回率', table_cell_left), Paragraph('0.856', table_cell_style)],
        [Paragraph('mAP50', table_cell_style), Paragraph('mAP@0.5', table_cell_left), Paragraph('0.874', table_cell_style)],
        [Paragraph('mAP50-95', table_cell_style), Paragraph('mAP@0.5:0.95', table_cell_left), Paragraph('0.623', table_cell_style)],
        [Paragraph('fps', table_cell_style), Paragraph('推理速度', table_cell_left), Paragraph('67.3', table_cell_style)],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(det_template, [AVAILABLE_WIDTH * 0.25, AVAILABLE_WIDTH * 0.35, AVAILABLE_WIDTH * 0.35]))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表9  检测实验记录模板', ParagraphStyle('Caption', fontName='SarasaRegular', fontSize=9, leading=14, textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')))
    story.append(Spacer(1, 12))
    
    story.append(h3('5.2  跟踪实验记录'))
    track_template = [
        [Paragraph('<b>字段</b>', table_header_style), Paragraph('<b>说明</b>', table_header_style), Paragraph('<b>示例值</b>', table_header_style)],
        [Paragraph('experiment_id', table_cell_style), Paragraph('实验编号', table_cell_left), Paragraph('Exp2-1', table_cell_style)],
        [Paragraph('algorithm', table_cell_style), Paragraph('跟踪算法', table_cell_left), Paragraph('ByteTrack', table_cell_style)],
        [Paragraph('MOTA', table_cell_style), Paragraph('多目标跟踪精度', table_cell_left), Paragraph('0.756', table_cell_style)],
        [Paragraph('IDF1', table_cell_style), Paragraph('ID F1分数', table_cell_left), Paragraph('0.812', table_cell_style)],
        [Paragraph('ID_Switch', table_cell_style), Paragraph('身份切换次数', table_cell_left), Paragraph('23', table_cell_style)],
        [Paragraph('MT', table_cell_style), Paragraph('主要跟踪目标数', table_cell_left), Paragraph('45', table_cell_style)],
        [Paragraph('ML', table_cell_style), Paragraph('主要丢失目标数', table_cell_left), Paragraph('3', table_cell_style)],
    ]
    story.append(Spacer(1, 6))
    story.append(make_table(track_template, [AVAILABLE_WIDTH * 0.25, AVAILABLE_WIDTH * 0.35, AVAILABLE_WIDTH * 0.35]))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表10  跟踪实验记录模板', ParagraphStyle('Caption', fontName='SarasaRegular', fontSize=9, leading=14, textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')))
    story.append(Spacer(1, 12))
    
    # ============================================================
    # 第6章：评估工具使用说明
    # ============================================================
    story.append(Spacer(1, 18))
    story.append(h1('6  评估工具使用说明'))
    story.append(Spacer(1, 12))
    
    story.append(body(
        '本项目提供了完整的评估工具链，支持各模块的独立评估和端到端综合评估。评估工具位于tools/目录下，'
        '运行脚本位于scripts/目录下。以下为各评估工具的使用方法。'
    ))
    
    story.append(h3('6.1  目标检测评估'))
    story.append(body_ni(
        '使用Ultralytics内置验证：python -c "from ultralytics import YOLO; model = YOLO(\'best.pt\'); model.val(data=\'classroom.yaml\')"'
    ))
    story.append(body_ni(
        '使用自定义评估脚本：python tools/eval_detection.py --model best.pt --data data/annotations/val.yaml --output outputs/eval_det.json'
    ))
    
    story.append(h3('6.2  多目标跟踪评估'))
    story.append(body_ni(
        '使用motmetrics评估：python tools/eval_tracking.py --gt data/annotations/gt_mot.txt --result outputs/tracking/result.txt'
    ))
    
    story.append(h3('6.3  人脸识别评估'))
    story.append(body_ni(
        'python tools/eval_face_recognition.py --db data/face_db --test data/face_db/test --output outputs/eval_face.json'
    ))
    
    story.append(h3('6.4  课堂状态分析评估'))
    story.append(body_ni(
        'python tools/eval_state.py --gt data/annotations/gt_states.json --pred outputs/pred_states.json --output outputs/eval_state.json'
    ))
    
    story.append(h3('6.5  完整评估'))
    story.append(body_ni(
        'python scripts/run_evaluation.py --task all --output outputs/evaluation'
    ))
    
    story.append(Spacer(1, 18))
    story.append(body(
        '以上评估工具均支持JSON格式的结果输出，便于后续的自动化分析和可视化。'
        '完整评估命令将依次运行所有模块的评估，并将结果汇总保存至指定输出目录。'
        '建议在每次实验后运行对应的评估工具，确保实验结果的完整记录和可追溯性。'
    ))
    
    # 构建文档
    doc.build(story)
    print(f'PDF文档已生成: {output_path}')
    return output_path


if __name__ == '__main__':
    build_document()
