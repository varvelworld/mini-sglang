from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    # running_reqs 从Set重构为Dict
    running_reqs: Dict[int, Req] = field(default_factory=dict)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        for req in reqs:
            self.running_reqs[req.uid] = req
        self.running_reqs = {uid: req for uid, req in self.running_reqs.items() if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.pop(req.uid, None)

    def abort_req(self, uid: int) -> Req | None:
        return self.running_reqs.pop(uid, None)

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs.values()) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs.values()), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
