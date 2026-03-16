from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.types import TASKS
from sqlmodel import delete, select

from agentchat.database.models.agent_graph_checkpoint import (
    AgentGraphCheckpointTable,
    AgentGraphWriteTable,
)
from agentchat.database.session import async_session_getter, session_getter


class MySQLCheckpointSaver(BaseCheckpointSaver[str]):
    """Minimal durable LangGraph checkpointer backed by the existing MySQL database."""

    def _build_config(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def _deserialize_checkpoint(self, row: AgentGraphCheckpointTable) -> Checkpoint:
        return self.serde.loads_typed((row.checkpoint_type, row.checkpoint_payload))

    def _deserialize_metadata(self, row: AgentGraphCheckpointTable) -> CheckpointMetadata:
        return self.serde.loads_typed((row.metadata_type, row.metadata_payload))

    def _deserialize_write(self, row: AgentGraphWriteTable) -> tuple[str, str, Any, str]:
        return row.task_id, row.channel, self.serde.loads_typed((row.payload_type, row.payload)), row.task_path

    def _build_tuple(
        self,
        row: AgentGraphCheckpointTable,
        writes: list[AgentGraphWriteTable],
        parent_writes: list[AgentGraphWriteTable],
        config: Optional[RunnableConfig] = None,
    ) -> CheckpointTuple:
        checkpoint = self._deserialize_checkpoint(row)
        pending_writes = [self._deserialize_write(write) for write in writes]
        pending_sends = [
            self.serde.loads_typed((write.payload_type, write.payload))
            for write in sorted(parent_writes, key=lambda item: (item.task_path, item.task_id, item.write_idx))
            if write.channel == TASKS
        ]

        checkpoint = {**checkpoint, "pending_sends": pending_sends}
        return CheckpointTuple(
            config=config or self._build_config(row.thread_id, row.checkpoint_ns, row.checkpoint_id),
            checkpoint=checkpoint,
            metadata=self._deserialize_metadata(row),
            parent_config=(
                self._build_config(row.thread_id, row.checkpoint_ns, row.parent_checkpoint_id)
                if row.parent_checkpoint_id
                else None
            ),
            pending_writes=[(task_id, channel, value) for task_id, channel, value, _ in pending_writes],
        )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        with session_getter() as session:
            statement = select(AgentGraphCheckpointTable).where(
                AgentGraphCheckpointTable.thread_id == thread_id,
                AgentGraphCheckpointTable.checkpoint_ns == checkpoint_ns,
            )
            if checkpoint_id:
                statement = statement.where(AgentGraphCheckpointTable.checkpoint_id == checkpoint_id)
            statement = statement.order_by(AgentGraphCheckpointTable.create_time.desc())
            row = session.exec(statement).first()
            if row is None:
                return None

            writes = session.exec(
                select(AgentGraphWriteTable).where(
                    AgentGraphWriteTable.thread_id == row.thread_id,
                    AgentGraphWriteTable.checkpoint_ns == row.checkpoint_ns,
                    AgentGraphWriteTable.checkpoint_id == row.checkpoint_id,
                )
            ).all()
            parent_writes = []
            if row.parent_checkpoint_id:
                parent_writes = session.exec(
                    select(AgentGraphWriteTable).where(
                        AgentGraphWriteTable.thread_id == row.thread_id,
                        AgentGraphWriteTable.checkpoint_ns == row.checkpoint_ns,
                        AgentGraphWriteTable.checkpoint_id == row.parent_checkpoint_id,
                    )
                ).all()
            return self._build_tuple(row, list(writes), list(parent_writes))

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        with session_getter() as session:
            statement = select(AgentGraphCheckpointTable)
            if config:
                statement = statement.where(
                    AgentGraphCheckpointTable.thread_id == config["configurable"]["thread_id"]
                )
                checkpoint_ns = config["configurable"].get("checkpoint_ns")
                if checkpoint_ns is not None:
                    statement = statement.where(AgentGraphCheckpointTable.checkpoint_ns == checkpoint_ns)
                if checkpoint_id := get_checkpoint_id(config):
                    statement = statement.where(AgentGraphCheckpointTable.checkpoint_id == checkpoint_id)
            if before and (before_checkpoint_id := get_checkpoint_id(before)):
                statement = statement.where(AgentGraphCheckpointTable.checkpoint_id < before_checkpoint_id)
            statement = statement.order_by(AgentGraphCheckpointTable.create_time.desc())
            rows = list(session.exec(statement).all())

            count = 0
            for row in rows:
                metadata = self._deserialize_metadata(row)
                if filter and not all(metadata.get(key) == value for key, value in filter.items()):
                    continue
                writes = session.exec(
                    select(AgentGraphWriteTable).where(
                        AgentGraphWriteTable.thread_id == row.thread_id,
                        AgentGraphWriteTable.checkpoint_ns == row.checkpoint_ns,
                        AgentGraphWriteTable.checkpoint_id == row.checkpoint_id,
                    )
                ).all()
                parent_writes = []
                if row.parent_checkpoint_id:
                    parent_writes = session.exec(
                        select(AgentGraphWriteTable).where(
                            AgentGraphWriteTable.thread_id == row.thread_id,
                            AgentGraphWriteTable.checkpoint_ns == row.checkpoint_ns,
                            AgentGraphWriteTable.checkpoint_id == row.parent_checkpoint_id,
                        )
                    ).all()
                yield self._build_tuple(
                    row=row,
                    writes=list(writes),
                    parent_writes=list(parent_writes),
                    config=self._build_config(row.thread_id, row.checkpoint_ns, row.checkpoint_id),
                )
                count += 1
                if limit is not None and count >= limit:
                    break

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        checkpoint_type, checkpoint_payload = self.serde.dumps_typed(checkpoint)
        metadata_type, metadata_payload = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))

        with session_getter() as session:
            row = AgentGraphCheckpointTable(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
                checkpoint_type=checkpoint_type,
                checkpoint_payload=checkpoint_payload,
                metadata_type=metadata_type,
                metadata_payload=metadata_payload,
                parent_checkpoint_id=config["configurable"].get("checkpoint_id"),
            )
            session.merge(row)
            session.commit()

        return self._build_config(thread_id, checkpoint_ns, checkpoint_id)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with session_getter() as session:
            for idx, (channel, value) in enumerate(writes):
                write_idx = WRITES_IDX_MAP.get(channel, idx)
                payload_type, payload = self.serde.dumps_typed(value)
                existing = session.exec(
                    select(AgentGraphWriteTable).where(
                        AgentGraphWriteTable.thread_id == thread_id,
                        AgentGraphWriteTable.checkpoint_ns == checkpoint_ns,
                        AgentGraphWriteTable.checkpoint_id == checkpoint_id,
                        AgentGraphWriteTable.task_id == task_id,
                        AgentGraphWriteTable.write_idx == write_idx,
                    )
                ).first()
                if existing and write_idx >= 0:
                    continue
                row = AgentGraphWriteTable(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    write_idx=write_idx,
                    channel=channel,
                    payload_type=payload_type,
                    payload=payload,
                    task_path=task_path,
                )
                session.merge(row)
            session.commit()

    def delete_thread(self, thread_id: str) -> None:
        with session_getter() as session:
            session.exec(delete(AgentGraphWriteTable).where(AgentGraphWriteTable.thread_id == thread_id))
            session.exec(delete(AgentGraphCheckpointTable).where(AgentGraphCheckpointTable.thread_id == thread_id))
            session.commit()

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        async with async_session_getter() as session:
            statement = select(AgentGraphCheckpointTable).where(
                AgentGraphCheckpointTable.thread_id == thread_id,
                AgentGraphCheckpointTable.checkpoint_ns == checkpoint_ns,
            )
            if checkpoint_id:
                statement = statement.where(AgentGraphCheckpointTable.checkpoint_id == checkpoint_id)
            statement = statement.order_by(AgentGraphCheckpointTable.create_time.desc())
            result = await session.exec(statement)
            row = result.first()
            if row is None:
                return None

            writes = await session.exec(
                select(AgentGraphWriteTable).where(
                    AgentGraphWriteTable.thread_id == row.thread_id,
                    AgentGraphWriteTable.checkpoint_ns == row.checkpoint_ns,
                    AgentGraphWriteTable.checkpoint_id == row.checkpoint_id,
                )
            )
            parent_writes = []
            if row.parent_checkpoint_id:
                parent_result = await session.exec(
                    select(AgentGraphWriteTable).where(
                        AgentGraphWriteTable.thread_id == row.thread_id,
                        AgentGraphWriteTable.checkpoint_ns == row.checkpoint_ns,
                        AgentGraphWriteTable.checkpoint_id == row.parent_checkpoint_id,
                    )
                )
                parent_writes = list(parent_result.all())
            return self._build_tuple(row, list(writes.all()), parent_writes)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config=config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config=config, checkpoint=checkpoint, metadata=metadata, new_versions=new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config=config, writes=writes, task_id=task_id, task_path=task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)
