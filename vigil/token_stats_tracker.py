import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TokenUsageRecord:
    def __init__(
        self,
        module: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model: str,
        timestamp: str | None = None,
    ):
        self.module = module
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.model = model
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "timestamp": self.timestamp,
        }


class TaskTokenStats:

    def __init__(self, task_id: str):
        """初始化任务统计

        Args:
            task_id: 任务ID
        """
        self.task_id = task_id
        self.records: list[TokenUsageRecord] = []
        self.module_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
        )

    def add_record(self, record: TokenUsageRecord) -> None:

        self.records.append(record)

        # 更新模块统计
        stats = self.module_stats[record.module]
        stats["prompt_tokens"] += record.prompt_tokens
        stats["completion_tokens"] += record.completion_tokens
        stats["total_tokens"] += record.total_tokens
        stats["call_count"] += 1

    def get_total_tokens(self) -> int:
        """获取任务的总 token 数量"""
        return sum(r.total_tokens for r in self.records)

    def get_module_tokens(self, module: str) -> int:

        return self.module_stats[module]["total_tokens"]

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "total_tokens": self.get_total_tokens(),
            "module_stats": dict(self.module_stats),
            "records": [r.to_dict() for r in self.records],
        }


class TokenStatsTracker:

    MODULE_INTENT_ANCHOR = "intent_anchor"
    MODULE_SPECULATIVE_REASONER = "speculative_reasoner"
    MODULE_NEURO_SYMBOLIC_VERIFIER = "neuro_symbolic_verifier"
    MODULE_PATH_CACHING = "path_caching"

    def __init__(self, output_dir: str = "./vigil_token_stats"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tasks: dict[str, TaskTokenStats] = {}
        self.current_task_id: str | None = None

        # 全局统计
        self.global_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}
        )

        logger.info(f"[TokenStatsTracker] Initialized with output_dir: {self.output_dir}")

    def start_task(self, task_id: str) -> None:

        self.current_task_id = task_id
        if task_id not in self.tasks:
            self.tasks[task_id] = TaskTokenStats(task_id)
            logger.debug(f"[TokenStatsTracker] Started tracking task: {task_id}")

    def record_usage(
        self,
        module: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model: str,
        task_id: str | None = None,
    ) -> None:

        record = TokenUsageRecord(
            module=module,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
        )

        # 确定任务ID
        target_task_id = task_id or self.current_task_id
        if target_task_id is None:
            logger.warning("[TokenStatsTracker] No task_id specified and no current task. Using 'unknown'.")
            target_task_id = "unknown"

        # 如果任务不存在，创建它
        if target_task_id not in self.tasks:
            self.tasks[target_task_id] = TaskTokenStats(target_task_id)

        # 添加到任务统计
        self.tasks[target_task_id].add_record(record)

        # 更新全局统计
        stats = self.global_stats[module]
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += completion_tokens
        stats["total_tokens"] += total_tokens
        stats["call_count"] += 1

        logger.debug(
            f"[TokenStatsTracker] Recorded usage for {module} in task {target_task_id}: "
            f"{total_tokens} tokens (prompt: {prompt_tokens}, completion: {completion_tokens})"
        )

    def end_task(self, task_id: str | None = None, save: bool = True) -> None:

        target_task_id = task_id or self.current_task_id
        if target_task_id is None:
            logger.warning("[TokenStatsTracker] No task_id specified and no current task.")
            return

        if target_task_id in self.tasks:
            task_stats = self.tasks[target_task_id]
            logger.info(
                f"[TokenStatsTracker] Task {target_task_id} completed. "
                f"Total tokens: {task_stats.get_total_tokens()}"
            )

            if save:
                self._save_task_stats(target_task_id)

        if self.current_task_id == target_task_id:
            self.current_task_id = None

    def _save_task_stats(self, task_id: str) -> None:

        if task_id not in self.tasks:
            logger.warning(f"[TokenStatsTracker] Task {task_id} not found.")
            return

        task_stats = self.tasks[task_id]

        # 保存为 JSON
        task_file = self.output_dir / f"task_{task_id}.json"
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task_stats.to_dict(), f, indent=2, ensure_ascii=False)

        logger.debug(f"[TokenStatsTracker] Saved task stats to {task_file}")

    def save_all_stats(self) -> None:

        for task_id in self.tasks:
            self._save_task_stats(task_id)

        # 2. 保存汇总统计
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(self.tasks),
            "global_stats": dict(self.global_stats),
            "total_tokens_all_tasks": sum(task.get_total_tokens() for task in self.tasks.values()),
            "module_breakdown": {
                module: {
                    "total_tokens": stats["total_tokens"],
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "call_count": stats["call_count"],
                    "avg_tokens_per_call": (
                        stats["total_tokens"] / stats["call_count"] if stats["call_count"] > 0 else 0
                    ),
                }
                for module, stats in self.global_stats.items()
            },
        }

        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"[TokenStatsTracker] Saved summary stats to {summary_file}")

        # 3. 保存为 CSV（便于后续分析）
        self._save_csv_report()

    def _save_csv_report(self) -> None:
        """保存 CSV 格式的报告"""
        import csv

        # 按任务的 CSV
        tasks_csv = self.output_dir / "tasks_summary.csv"
        with open(tasks_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "task_id",
                    "total_tokens",
                    "intent_anchor_tokens",
                    "speculative_reasoner_tokens",
                    "neuro_symbolic_verifier_tokens",
                    "path_caching_tokens",
                ]
            )

            for task_id, task_stats in self.tasks.items():
                writer.writerow(
                    [
                        task_id,
                        task_stats.get_total_tokens(),
                        task_stats.get_module_tokens(self.MODULE_INTENT_ANCHOR),
                        task_stats.get_module_tokens(self.MODULE_SPECULATIVE_REASONER),
                        task_stats.get_module_tokens(self.MODULE_NEURO_SYMBOLIC_VERIFIER),
                        task_stats.get_module_tokens(self.MODULE_PATH_CACHING),
                    ]
                )

        logger.info(f"[TokenStatsTracker] Saved CSV report to {tasks_csv}")

        # 按模块的 CSV
        modules_csv = self.output_dir / "modules_summary.csv"
        with open(modules_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "module",
                    "total_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                    "call_count",
                    "avg_tokens_per_call",
                ]
            )

            for module, stats in self.global_stats.items():
                avg = stats["total_tokens"] / stats["call_count"] if stats["call_count"] > 0 else 0
                writer.writerow(
                    [
                        module,
                        stats["total_tokens"],
                        stats["prompt_tokens"],
                        stats["completion_tokens"],
                        stats["call_count"],
                        f"{avg:.2f}",
                    ]
                )

        logger.info(f"[TokenStatsTracker] Saved modules CSV to {modules_csv}")

    def get_task_stats(self, task_id: str) -> TaskTokenStats | None:
        """获取任务的统计信息

        Args:
            task_id: 任务ID

        Returns:
            任务统计对象，如果不存在返回 None
        """
        return self.tasks.get(task_id)

    def get_summary(self) -> dict[str, Any]:
        """获取汇总统计

        Returns:
            汇总统计字典
        """
        return {
            "total_tasks": len(self.tasks),
            "total_tokens": sum(task.get_total_tokens() for task in self.tasks.values()),
            "module_stats": dict(self.global_stats),
        }

    def reset(self) -> None:
        """重置所有统计"""
        self.tasks.clear()
        self.global_stats.clear()
        self.current_task_id = None
        logger.info("[TokenStatsTracker] Reset all statistics")


# 全局单例实例
_global_tracker: TokenStatsTracker | None = None


def get_global_tracker(output_dir: str = "./vigil_token_stats") -> TokenStatsTracker:
    """获取全局 token 统计追踪器

    Args:
        output_dir: 输出目录

    Returns:
        全局追踪器实例
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TokenStatsTracker(output_dir=output_dir)
    return _global_tracker


def reset_global_tracker() -> None:
    """重置全局追踪器"""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()
    _global_tracker = None
