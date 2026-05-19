"""
时序融合与状态平滑模块

对逐帧的状态分类结果进行时序融合，消除短时抖动，
输出稳定的课堂状态统计。

核心方法:
    1. 滑动窗口投票: 在时间窗口内对状态进行多数投票
    2. 状态机平滑: 基于状态转移约束消除不合理跳变
    3. 统计聚合: 输出到课率、抬头率、离座次数等统计指标

状态转移约束:
    - 专注 -> 分心: 至少连续3帧分心才确认
    - 专注 -> 低头: 至少连续5帧低头才确认
    - 低头 -> 专注: 至少连续3帧专注才确认
    - 任何状态 -> 离座: 至少连续10帧丢失才确认
    - 离座 -> 任何状态: 重新检测到即恢复
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .state_classifier import StudentState, StateResult


@dataclass
class SmoothedState:
    """平滑后的状态"""
    track_id: int
    state: StudentState
    confidence: float
    identity: Optional[str] = None
    duration: float = 0.0  # 当前状态持续时间(秒)


@dataclass
class ClassroomStatistics:
    """课堂统计信息"""
    total_students: int           # 到课人数
    focused_count: int            # 专注人数
    distracted_count: int         # 分心人数
    head_down_count: int          # 低头人数
    left_seat_count: int          # 离座人数
    hand_raising_count: int       # 举手人数
    focus_rate: float             # 抬头率/专注率
    attendance_rate: float        # 到课率
    timestamp: float              # 时间戳


class TemporalFusion:
    """
    时序融合器
    
    对逐帧状态分类结果进行时序平滑和统计聚合。
    
    Args:
        window_size: 滑动窗口大小（帧数）
        fps: 视频帧率
        state_transition_frames: 状态转移所需最少连续帧数
        leave_seat_threshold: 离座判定帧数阈值
    """

    # 状态转移所需最少连续帧数
    DEFAULT_TRANSITION_FRAMES = {
        (StudentState.FOCUSED, StudentState.DISTRACTED): 3,
        (StudentState.FOCUSED, StudentState.HEAD_DOWN): 5,
        (StudentState.HEAD_DOWN, StudentState.FOCUSED): 3,
        (StudentState.DISTRACTED, StudentState.FOCUSED): 3,
        (StudentState.HEAD_DOWN, StudentState.DISTRACTED): 3,
        (StudentState.DISTRACTED, StudentState.HEAD_DOWN): 3,
    }

    def __init__(
        self,
        window_size: int = 15,
        fps: float = 30.0,
        state_transition_frames: Optional[Dict[Tuple[StudentState, StudentState], int]] = None,
        leave_seat_threshold: int = 30,
    ):
        self.window_size = window_size
        self.fps = fps
        self.leave_seat_threshold = leave_seat_threshold

        self.transition_frames = state_transition_frames or self.DEFAULT_TRANSITION_FRAMES

        # 每个跟踪目标的历史状态队列
        self._state_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        # 每个跟踪目标的当前确认状态
        self._confirmed_state: Dict[int, StudentState] = {}
        # 每个跟踪目标的状态候选计数
        self._candidate_count: Dict[int, Dict[StudentState, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # 每个跟踪目标的身份信息
        self._identities: Dict[int, str] = {}
        # 离座计数
        self._lost_count: Dict[int, int] = defaultdict(int)
        # 离座事件记录
        self._leave_events: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        # 当前离座起始时间
        self._leave_start: Dict[int, float] = {}

        # 课堂级别统计历史
        self._stats_history: List[ClassroomStatistics] = []

    def update(
        self,
        state_results: List[StateResult],
        frame_id: int,
        timestamp: float,
        registered_students: int = 0,
    ) -> Tuple[List[SmoothedState], ClassroomStatistics]:
        """
        更新时序融合，输出平滑状态和课堂统计
        
        Args:
            state_results: 当前帧的状态分类结果
            frame_id: 当前帧ID
            timestamp: 当前时间戳
            registered_students: 注册学生总数（用于计算到课率）
            
        Returns:
            (平滑状态列表, 课堂统计)
        """
        current_track_ids = set()

        for result in state_results:
            tid = result.track_id
            current_track_ids.add(tid)

            # 记录身份
            if result.identity is not None:
                self._identities[tid] = result.identity

            # 更新状态历史
            self._state_history[tid].append(result.state)

            # 状态转移平滑
            smoothed_state = self._smooth_state(tid, result.state)
            self._confirmed_state[tid] = smoothed_state

            # 如果目标在跟踪中，重置离座计数
            if result.state != StudentState.LEFT_SEAT:
                self._lost_count[tid] = 0
                if tid in self._leave_start:
                    # 离座结束
                    self._leave_events[tid].append(
                        (self._leave_start[tid], timestamp)
                    )
                    del self._leave_start[tid]

        # 处理离座目标
        for tid in list(self._lost_count.keys()):
            if tid not in current_track_ids:
                self._lost_count[tid] += 1
                if self._lost_count[tid] >= self.leave_seat_threshold:
                    self._confirmed_state[tid] = StudentState.LEFT_SEAT
                    if tid not in self._leave_start:
                        self._leave_start[tid] = timestamp

        # 生成平滑状态列表
        smoothed_states = []
        for tid, state in self._confirmed_state.items():
            # 计算状态持续时间
            history = self._state_history[tid]
            duration = 0.0
            if len(history) > 1:
                duration = len(history) / self.fps

            smoothed_states.append(SmoothedState(
                track_id=tid,
                state=state,
                confidence=0.8,
                identity=self._identities.get(tid),
                duration=duration,
            ))

        # 计算课堂统计
        stats = self._compute_statistics(
            smoothed_states, timestamp, registered_students
        )
        self._stats_history.append(stats)

        return smoothed_states, stats

    def _smooth_state(self, track_id: int, raw_state: StudentState) -> StudentState:
        """
        状态转移平滑
        
        基于状态转移约束，只有当候选状态连续出现足够帧数时
        才确认状态转移，避免短时抖动。
        """
        current = self._confirmed_state.get(track_id, raw_state)

        if current == raw_state:
            # 状态未变化，重置候选计数
            self._candidate_count[track_id] = defaultdict(int)
            return current

        # 状态变化，增加候选计数
        self._candidate_count[track_id][raw_state] += 1

        # 检查是否满足转移条件
        transition_key = (current, raw_state)
        required_frames = self.transition_frames.get(transition_key, 2)

        if self._candidate_count[track_id][raw_state] >= required_frames:
            # 确认状态转移
            self._candidate_count[track_id] = defaultdict(int)
            return raw_state

        # 未满足转移条件，保持当前状态
        return current

    def _compute_statistics(
        self,
        smoothed_states: List[SmoothedState],
        timestamp: float,
        registered_students: int,
    ) -> ClassroomStatistics:
        """计算课堂统计信息"""
        state_counts = defaultdict(int)
        for s in smoothed_states:
            state_counts[s.state] += 1

        total = len(smoothed_states)
        focused = state_counts.get(StudentState.FOCUSED, 0)
        distracted = state_counts.get(StudentState.DISTRACTED, 0)
        head_down = state_counts.get(StudentState.HEAD_DOWN, 0)
        left_seat = state_counts.get(StudentState.LEFT_SEAT, 0)
        hand_raising = state_counts.get(StudentState.HAND_RAISING, 0)

        # 抬头率 = (专注 + 分心 + 举手) / 在座人数
        in_seat = total - left_seat
        focus_rate = (focused + distracted + hand_raising) / in_seat if in_seat > 0 else 0.0

        # 到课率 = 在座人数 / 注册人数
        attendance_rate = in_seat / registered_students if registered_students > 0 else (1.0 if total > 0 else 0.0)

        return ClassroomStatistics(
            total_students=total,
            focused_count=focused,
            distracted_count=distracted,
            head_down_count=head_down,
            left_seat_count=left_seat,
            hand_raising_count=hand_raising,
            focus_rate=focus_rate,
            attendance_rate=attendance_rate,
            timestamp=timestamp,
        )

    def get_leave_seat_events(self) -> Dict[int, List[Tuple[float, float]]]:
        """获取所有离座事件（起止时间）"""
        return dict(self._leave_events)

    def get_statistics_history(self) -> List[ClassroomStatistics]:
        """获取统计历史"""
        return self._stats_history

    def reset(self):
        """重置所有状态"""
        self._state_history.clear()
        self._confirmed_state.clear()
        self._candidate_count.clear()
        self._identities.clear()
        self._lost_count.clear()
        self._leave_events.clear()
        self._leave_start.clear()
        self._stats_history.clear()
