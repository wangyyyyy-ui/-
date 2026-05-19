"""
统计导出模块

将课堂分析结果导出为多种格式:
    - CSV: 时序统计数据
    - JSON: 结构化分析报告
    - 图表: 趋势图、分布图
"""

import csv
import json
import os
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime

from ..state_analysis.temporal_fusion import ClassroomStatistics


class StatisticsExporter:
    """
    统计数据导出器
    
    支持将课堂统计数据导出为CSV、JSON格式，
    并生成趋势分析图表。
    
    Args:
        output_dir: 输出目录
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(
        self,
        stats_history: List[ClassroomStatistics],
        filename: Optional[str] = None,
    ) -> str:
        """
        导出统计历史为CSV文件
        
        Args:
            stats_history: 统计历史列表
            filename: 输出文件名，默认自动生成
            
        Returns:
            输出文件路径
        """
        if filename is None:
            filename = f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = self.output_dir / filename

        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "时间戳", "到课人数", "专注人数", "分心人数",
                "低头人数", "离座人数", "举手人数", "抬头率", "到课率",
            ])

            for stats in stats_history:
                writer.writerow([
                    f"{stats.timestamp:.2f}",
                    stats.total_students - stats.left_seat_count,
                    stats.focused_count,
                    stats.distracted_count,
                    stats.head_down_count,
                    stats.left_seat_count,
                    stats.hand_raising_count,
                    f"{stats.focus_rate:.4f}",
                    f"{stats.attendance_rate:.4f}",
                ])

        print(f"[StatisticsExporter] CSV导出: {filepath}")
        return str(filepath)

    def export_json(
        self,
        stats_history: List[ClassroomStatistics],
        leave_events: Dict,
        filename: Optional[str] = None,
    ) -> str:
        """
        导出分析报告为JSON文件
        
        Args:
            stats_history: 统计历史列表
            leave_events: 离座事件记录
            filename: 输出文件名
            
        Returns:
            输出文件路径
        """
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.output_dir / filename

        # 计算汇总统计
        if len(stats_history) > 0:
            avg_focus_rate = np.mean([s.focus_rate for s in stats_history])
            avg_attendance = np.mean([s.attendance_rate for s in stats_history])
            max_left_seat = max(s.left_seat_count for s in stats_history)
        else:
            avg_focus_rate = 0.0
            avg_attendance = 0.0
            max_left_seat = 0

        report = {
            "report_time": datetime.now().isoformat(),
            "summary": {
                "average_focus_rate": round(avg_focus_rate, 4),
                "average_attendance_rate": round(avg_attendance, 4),
                "max_left_seat_count": max_left_seat,
                "total_leave_events": sum(len(v) for v in leave_events.values()),
            },
            "time_series": [
                {
                    "timestamp": round(s.timestamp, 2),
                    "total_students": s.total_students,
                    "focused": s.focused_count,
                    "distracted": s.distracted_count,
                    "head_down": s.head_down_count,
                    "left_seat": s.left_seat_count,
                    "hand_raising": s.hand_raising_count,
                    "focus_rate": round(s.focus_rate, 4),
                    "attendance_rate": round(s.attendance_rate, 4),
                }
                for s in stats_history
            ],
            "leave_seat_events": {
                str(tid): [
                    {"start": round(start, 2), "end": round(end, 2)}
                    for start, end in events
                ]
                for tid, events in leave_events.items()
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[StatisticsExporter] JSON导出: {filepath}")
        return str(filepath)

    def export_plots(
        self,
        stats_history: List[ClassroomStatistics],
        filename_prefix: Optional[str] = None,
    ) -> List[str]:
        """
        生成统计趋势图
        
        Args:
            stats_history: 统计历史列表
            filename_prefix: 文件名前缀
            
        Returns:
            输出文件路径列表
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            matplotlib.font_manager.fontManager.addfont('/usr/share/fonts/truetype/chinese/SimHei.ttf')
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError:
            print("[StatisticsExporter] matplotlib未安装，跳过图表生成")
            return []

        if filename_prefix is None:
            filename_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

        output_files = []

        if len(stats_history) == 0:
            return output_files

        timestamps = [s.timestamp for s in stats_history]
        focus_rates = [s.focus_rate * 100 for s in stats_history]
        attendance_rates = [s.attendance_rate * 100 for s in stats_history]
        focused_counts = [s.focused_count for s in stats_history]
        head_down_counts = [s.head_down_count for s in stats_history]
        distracted_counts = [s.distracted_count for s in stats_history]

        # 图1: 抬头率与到课率趋势
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(timestamps, focus_rates, label="抬头率(%)", color="green", linewidth=2)
        ax.plot(timestamps, attendance_rates, label="到课率(%)", color="blue", linewidth=2)
        ax.set_xlabel("时间 (秒)")
        ax.set_ylabel("百分比 (%)")
        ax.set_title("课堂抬头率与到课率趋势")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path1 = str(self.output_dir / f"{filename_prefix}_rates.png")
        fig.savefig(path1, dpi=150)
        plt.close(fig)
        output_files.append(path1)

        # 图2: 各状态人数趋势
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(timestamps, focused_counts, label="专注", color="green", linewidth=2)
        ax.plot(timestamps, head_down_counts, label="低头", color="red", linewidth=2)
        ax.plot(timestamps, distracted_counts, label="分心", color="orange", linewidth=2)
        ax.set_xlabel("时间 (秒)")
        ax.set_ylabel("人数")
        ax.set_title("课堂各状态人数变化趋势")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path2 = str(self.output_dir / f"{filename_prefix}_counts.png")
        fig.savefig(path2, dpi=150)
        plt.close(fig)
        output_files.append(path2)

        print(f"[StatisticsExporter] 图表导出完成: {len(output_files)}个文件")
        return output_files
