import logging
from typing import Any

from vigil.config import VIGILConfig

logger = logging.getLogger(__name__)


class VerifiedPath:
    def __init__(self, tool_name: str, execution_count: int = 1):
        self.tool_name = tool_name
        self.execution_count = execution_count


class PathCache:
    
    def __init__(
        self,
        config: VIGILConfig,
        openai_client=None,
        token_tracker=None,
    ):
        self.config = config
        # 核心存储：{abstract_step_description: tool_name}
        self._cache: dict[str, str] = {}
        # 统计信息：{abstract_step_description: execution_count}
        self._execution_counts: dict[str, int] = {}

    def add_verified_path(
        self,
        user_query: str,
        tool_name: str,
        arguments: dict[str, Any],
        outcome: str,
        step_index: int | None = None,
        abstract_step: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # 只缓存成功的执行，且必须有 abstract_step_description
        if outcome != "success" or not abstract_step:
            return

        # 规范化 abstract step description
        normalized_key = self._normalize_key(abstract_step)

        # 简单的字典写入操作: dict[key] = value
        self._cache[normalized_key] = tool_name

        # 更新执行计数
        if normalized_key in self._execution_counts:
            self._execution_counts[normalized_key] += 1
        else:
            self._execution_counts[normalized_key] = 1

        logger.info(
            f"[PathCache] Cached: '{abstract_step}' -> '{tool_name}' "
            f"(count: {self._execution_counts[normalized_key]})"
        )

    def retrieve_paths_by_abstract_step(
        self, abstract_step_step_type: str, top_k: int = 3
    ) -> list[VerifiedPath]:
        # 规范化 key
        normalized_key = self._normalize_key(abstract_step_step_type)

        # 简单的字典读取操作: dict.get(key)
        tool_name = self._cache.get(normalized_key)

        if tool_name:
            # 找到缓存，返回 VerifiedPath 对象
            execution_count = self._execution_counts.get(normalized_key, 1)
            verified_path = VerifiedPath(tool_name, execution_count)

            logger.info(
                f"[PathCache] ✓ Cache HIT: '{abstract_step_step_type}...' -> '{tool_name}' "
                f"(used {execution_count} times)"
            )
            return [verified_path]
        else:
            # 未找到缓存
            logger.debug(
                f"[PathCache] ✗ Cache MISS: No cached tool for '{abstract_step_step_type}...'"
            )
            return []

    def select_tool_with_llm(
        self,
        abstract_step_description: str,
        candidate_paths: list[VerifiedPath],
    ) -> tuple[str | None, str | None]:
        
        if not candidate_paths:
            return None, None

        # 简化版本：直接返回第一个（也是唯一的）候选工具
        selected_path = candidate_paths[0]
        tool_name = selected_path.tool_name
        rationale = f"Cached tool from previous successful execution (used {selected_path.execution_count} times)"

        return tool_name, rationale

    def _normalize_key(self, abstract_step: str) -> str:
       
        return " ".join(abstract_step.lower().split())

    def get_stats(self) -> dict[str, Any]:

        total_paths = len(self._cache)
        total_executions = sum(self._execution_counts.values())

        return {
            "total_cached_paths": total_paths,
            "successful_paths": total_paths,  # 简化版本只缓存成功的
            "failed_paths": 0,  # 简化版本不缓存失败的
            "total_executions": total_executions,
            "unique_queries": total_paths,
        }

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._execution_counts.clear()
        logger.info("[PathCache] Cache cleared")

    def export_cache(self) -> dict[str, Any]:
        
        return {
            "cache": self._cache,
            "execution_counts": self._execution_counts,
            "stats": self.get_stats(),
        }

    def import_cache(self, cache_data: dict[str, Any]) -> None:

        self.clear()

        self._cache = cache_data.get("cache", {})
        self._execution_counts = cache_data.get("execution_counts", {})

        logger.info(f"[PathCache] Imported {len(self._cache)} paths from cache data")

    # ===== 向后兼容的方法 =====
    # 这些方法保留是为了不破坏现有代码的调用

    def retrieve_paths(
        self, user_query: str, step_index: int | None = None
    ) -> list[VerifiedPath]:
        """向后兼容方法 - 转发到 retrieve_paths_by_abstract_step"""
        return self.retrieve_paths_by_abstract_step(user_query, top_k=1)

    def get_recommended_tool(
        self, user_query: str, step_index: int | None = None
    ) -> str | None:
        """向后兼容方法 - 获取推荐的工具"""
        paths = self.retrieve_paths_by_abstract_step(user_query, top_k=1)
        if paths:
            return paths[0].tool_name
        return None
