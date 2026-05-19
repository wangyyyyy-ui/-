"""
课堂状态分类器

综合头部姿态、视线方向、身体姿态等多模态信息，
对每个学生的课堂学习状态进行分类判断。

课堂状态定义:
    - 专注 (focused): 正视前方/看屏幕，正常坐姿
    - 分心 (distracted): 侧转/视线偏离，但仍在座位上
    - 低头 (head_down): 低头看桌面，可能看手机/书本
    - 离座 (left_seat): 不在座位上
    - 举手 (hand_raising): 举手状态
    - 未知 (unknown): 无法判断

分类策略:
    采用规则引擎 + 可选轻量级分类器的混合方案:
    1. 规则引擎: 基于头部姿态和身体姿态的阈值判断
    2. 轻量分类器: 基于时序特征的MLP分类（可选）
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from ..pose_estimation.head_pose import HeadDirection, GazeDirection, HeadPoseResult
from ..pose_estimation.body_pose import BodyAction, BodyPoseResult


class StudentState(Enum):
    """学生课堂状态枚举"""
    FOCUSED = "focused"           # 专注听课
    DISTRACTED = "distracted"     # 分心
    HEAD_DOWN = "head_down"       # 低头
    LEFT_SEAT = "left_seat"       # 离座
    HAND_RAISING = "hand_raising" # 举手
    UNKNOWN = "unknown"           # 未知


@dataclass
class StateResult:
    """状态分类结果"""
    track_id: int
    state: StudentState
    confidence: float
    head_direction: Optional[HeadDirection] = None
    gaze_direction: Optional[GazeDirection] = None
    body_action: Optional[BodyAction] = None
    identity: Optional[str] = None


class StateClassifier:
    """
    课堂状态分类器
    
    基于多模态信息融合的学生课堂状态判断。
    采用分层规则引擎，优先级从高到低:
        1. 离座检测（跟踪丢失）
        2. 举手检测（身体姿态）
        3. 低头检测（头部姿态）
        4. 分心检测（视线方向）
        5. 专注状态（默认）
    
    Args:
        head_down_pitch_thresh: 低头判定pitch阈值（度）
        distracted_yaw_thresh: 分心判定yaw阈值（度）
        track_lost_frames: 跟踪丢失多少帧判定为离座
    """

    def __init__(
        self,
        head_down_pitch_thresh: float = 15.0,
        distracted_yaw_thresh: float = 30.0,
        track_lost_frames: int = 30,
    ):
        self.head_down_pitch_thresh = head_down_pitch_thresh
        self.distracted_yaw_thresh = distracted_yaw_thresh
        self.track_lost_frames = track_lost_frames

    def classify(
        self,
        track_id: int,
        head_pose: Optional[HeadPoseResult] = None,
        body_pose: Optional[BodyPoseResult] = None,
        is_tracked: bool = True,
        identity: Optional[str] = None,
    ) -> StateResult:
        """
        对单个学生进行状态分类
        
        Args:
            track_id: 跟踪ID
            head_pose: 头部姿态结果
            body_pose: 身体姿态结果
            is_tracked: 是否仍在跟踪中
            identity: 身份标签
            
        Returns:
            状态分类结果
        """
        # 优先级1: 离座检测
        if not is_tracked:
            return StateResult(
                track_id=track_id,
                state=StudentState.LEFT_SEAT,
                confidence=0.9,
                identity=identity,
            )

        # 优先级2: 举手检测
        if body_pose is not None and body_pose.action == BodyAction.HAND_RAISING:
            return StateResult(
                track_id=track_id,
                state=StudentState.HAND_RAISING,
                confidence=body_pose.confidence,
                head_direction=head_pose.head_direction if head_pose else None,
                gaze_direction=head_pose.gaze_direction if head_pose else None,
                body_action=body_pose.action,
                identity=identity,
            )

        # 优先级3: 低头检测
        if head_pose is not None:
            if head_pose.head_direction == HeadDirection.DOWN:
                return StateResult(
                    track_id=track_id,
                    state=StudentState.HEAD_DOWN,
                    confidence=head_pose.confidence,
                    head_direction=head_pose.head_direction,
                    gaze_direction=head_pose.gaze_direction,
                    body_action=body_pose.action if body_pose else None,
                    identity=identity,
                )

            # 优先级4: 分心检测
            if head_pose.head_direction in [HeadDirection.LEFT, HeadDirection.RIGHT]:
                return StateResult(
                    track_id=track_id,
                    state=StudentState.DISTRACTED,
                    confidence=head_pose.confidence,
                    head_direction=head_pose.head_direction,
                    gaze_direction=head_pose.gaze_direction,
                    body_action=body_pose.action if body_pose else None,
                    identity=identity,
                )

            # 优先级5: 专注状态
            if head_pose.head_direction == HeadDirection.FORWARD:
                return StateResult(
                    track_id=track_id,
                    state=StudentState.FOCUSED,
                    confidence=head_pose.confidence,
                    head_direction=head_pose.head_direction,
                    gaze_direction=head_pose.gaze_direction,
                    body_action=body_pose.action if body_pose else None,
                    identity=identity,
                )

        # 无头部姿态信息时，根据身体姿态判断
        if body_pose is not None:
            if body_pose.action == BodyAction.LEANING:
                return StateResult(
                    track_id=track_id,
                    state=StudentState.HEAD_DOWN,
                    confidence=body_pose.confidence * 0.7,
                    body_action=body_pose.action,
                    identity=identity,
                )
            elif body_pose.action == BodyAction.SITTING:
                return StateResult(
                    track_id=track_id,
                    state=StudentState.FOCUSED,
                    confidence=0.5,
                    body_action=body_pose.action,
                    identity=identity,
                )

        return StateResult(
            track_id=track_id,
            state=StudentState.UNKNOWN,
            confidence=0.0,
            identity=identity,
        )

    def classify_batch(
        self,
        tracks: list,
        head_poses: Dict[int, HeadPoseResult],
        body_poses: Dict[int, BodyPoseResult],
        lost_track_ids: List[int],
        identities: Dict[int, str],
    ) -> List[StateResult]:
        """
        批量状态分类
        
        Args:
            tracks: 当前帧的跟踪目标列表
            head_poses: {track_id: HeadPoseResult}
            body_poses: {track_id: BodyPoseResult}
            lost_track_ids: 丢失的跟踪ID列表
            identities: {track_id: identity_name}
            
        Returns:
            状态分类结果列表
        """
        results = []

        # 处理仍在跟踪中的目标
        for track in tracks:
            tid = track.track_id
            results.append(self.classify(
                track_id=tid,
                head_pose=head_poses.get(tid),
                body_pose=body_poses.get(tid),
                is_tracked=True,
                identity=identities.get(tid),
            ))

        # 处理离座目标
        for tid in lost_track_ids:
            results.append(self.classify(
                track_id=tid,
                is_tracked=False,
                identity=identities.get(tid),
            ))

        return results
