from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Protocol, runtime_checkable

from minisgl.core import Batch, Req


@runtime_checkable
class DecodeHook(Protocol):
    def on_filter(
        self,
        added_uids: list[int],
        removed_uids: list[int],
        running: Dict[int, Req],
    ) -> None:
        """filter_reqs 完成后：新加入、被移除的 uid 列表，以及当前 running_reqs。"""
        ...

    def on_abort(self, uid: int, req: Req | None) -> None:
        """abort_req 完成后：目标 uid，命中时为 Req，未命中时为 None。"""
        ...

    def on_remove(self, req: Req) -> None:
        """remove_req 完成后：被移除的 Req。"""
        ...

    def on_schedule(self, batch: Batch | None) -> None:
        """schedule_next_batch 完成后：生成的 Batch，队列为空时为 None。"""
        ...


@dataclass
class DecodeManager:
    page_size: int
    # running_reqs 从Set重构为Dict
    running_reqs: Dict[int, Req] = field(default_factory=dict)

    def filter_reqs(self, reqs: Iterable[Req], hook: DecodeHook | None = None) -> None:
        added_uids = []
        for req in reqs:
            self.running_reqs[req.uid] = req
            added_uids.append(req.uid)
        removed_uids = [
            uid for uid, req in self.running_reqs.items() if not req.can_decode
        ]
        self.running_reqs = {uid: req for uid, req in self.running_reqs.items() if req.can_decode}
        if hook is not None:
            hook.on_filter(added_uids, removed_uids, self.running_reqs)

    def remove_req(self, req: Req, hook: DecodeHook | None = None) -> None:
        self.running_reqs.pop(req.uid, None)
        if hook is not None:
            hook.on_remove(req)

    def abort_req(self, uid: int, hook: DecodeHook | None = None) -> Req | None:
        req = self.running_reqs.pop(uid, None)
        if hook is not None:
            hook.on_abort(uid, req)
        return req

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs.values()) + tokens_reserved

    def schedule_next_batch(self, hook: DecodeHook | None = None) -> Batch | None:
        if not self.runnable:
            if hook is not None:
                hook.on_schedule(None)
            return None
        batch = Batch(reqs=list(self.running_reqs.values()), phase="decode")
        if hook is not None:
            hook.on_schedule(batch)
        return batch

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
