"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import copy
import io
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.configuration import SETTINGS
from raychat.packages import Manifest
from raychat.type_support import override
from raychat.validation import array_field, json_object, object_field, text_field
from tests.assertions import TypedTestCase
from tests.plugin_support import ScriptedChat, plugin_module, registered_run
from tests.transport_support import captured

if TYPE_CHECKING:
    from collections.abc import Mapping

    from plugins import context as _rc_context
    from plugins import memory as _rc_memory
    from plugins import skills as _rc_skills
    from raychat.sdk import Action, EventCallback
else:
    _rc_context = plugin_module("context")
    _rc_memory = plugin_module("memory")
    _rc_skills = plugin_module("skills")


def _manifest_instructions(name: str) -> str:
    raw: object = getattr(plugin_module(name), "__plugin_manifest__", None)
    if not isinstance(raw, Manifest):
        message = "Captured plugin module has no concrete manifest."
        raise TypeError(message)
    return raw.instructions


def _decode_object(data: str) -> dict[str, object]:
    return object_field(json_object(data), "decoded host result")


def _json_dump(
    value: object,
    *,
    ensure_ascii: bool = True,
    separators: tuple[str, str] | None = None,
) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, separators=separators)


def _record_events(events: list[tuple[str, Mapping[str, object]]]) -> EventCallback:
    def record(name: str, payload: Mapping[str, object]) -> None:
        events.append((name, payload))

    return record


def _quiet_event(_name: str, _payload: Mapping[str, object]) -> None:
    pass


def _result(payload: Mapping[str, object]) -> dict[str, object]:
    return object_field(payload["result"], "event result")


def _compaction_body_size(probe: ScriptedChat[str], role: str, budget: int) -> int:
    pinned_count = 1 if role == "user" else 2
    bare_pinned = copy.deepcopy(probe.calls[1][:pinned_count])
    marked_pinned = copy.deepcopy(bare_pinned)
    anchor_index = 0 if role == "user" else 1
    marked_pinned[anchor_index]["content"] += (
        _rc_context.COMPACTION_SEPARATOR + _rc_context.COMPACTION_PREFIX
    )
    headroom = _rc_context.messages_size(marked_pinned) - _rc_context.messages_size(
        bare_pinned,
    )
    return budget - _rc_context.messages_size(bare_pinned) - headroom + 1


class _RunFixture(TypedTestCase):
    """Provide an isolated workspace for each composed plugin conversation."""

    @override
    def setUp(self) -> None:
        """Create the temporary conversation workspace."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    @override
    def tearDown(self) -> None:
        """Release the conversation workspace."""
        self.temporary.cleanup()


class RunAgentTests(_RunFixture):
    """Check instruction, context, memory and skill behavior in real sessions."""

    def test_done_returns_message_and_initial_context_has_catalog_and_memory(
        self,
    ) -> None:
        """Check done returns message and initial context has catalog and memory."""
        skill = _rc_skills.Skill("test", "Testing", "Do tests", self.root / "SKILL.md")
        skills = _rc_skills.SkillStore([skill])
        memory = _rc_memory.MemoryStore(self.root / "memory.json")
        memory.add("Use UTF-8")
        chat = ScriptedChat(['{"action":"done","message":"complete"}'])

        result = registered_run(
            chat,
            "build it",
            self.root / "workspace",
            skills=skills,
            memory=memory,
        )

        self.equal(result, "complete")
        self.equal(len(chat.calls), 1)
        system = chat.calls[0][0]["content"]
        self.require(('"name": "test"') in (system))
        self.require(("Use UTF-8") in (system))
        self.equal(chat.calls[0][1], {"role": "user", "content": "build it"})

    def test_protocol_override_is_per_run_and_default_remains_exact(self) -> None:
        """Check protocol override is per run and default remains exact."""
        custom = "CUSTOM OPTIMIZER CANDIDATE\n"
        custom_chat = ScriptedChat(
            [
                '{"action":"list","path":"."}',
                '{"action":"done","message":"custom"}',
            ],
        )
        default_chat = ScriptedChat(['{"action":"done","message":"default"}'])

        registered_run(custom_chat, "task", self.root / "custom", protocol=custom)
        registered_run(default_chat, "task", self.root / "default")

        self.equal(len(custom_chat.calls), 2)
        for call in custom_chat.calls:
            self.require(call[0]["content"].startswith(custom))
            self.require(
                (
                    text_field(
                        SETTINGS.plugins.settings["context"]["protocol"],
                        "context.protocol",
                    )
                )
                not in (call[0]["content"]),
            )
            self.require(
                (_manifest_instructions("filesystem")) in (call[0]["content"]),
            )
        self.require(
            default_chat.calls[0][0]["content"].startswith(
                text_field(
                    SETTINGS.plugins.settings["context"]["protocol"],
                    "context.protocol",
                ),
            ),
        )
        self.require(
            (_manifest_instructions("filesystem"))
            in (default_chat.calls[0][0]["content"]),
        )

    def test_all_instruction_roles_keep_dynamic_context_and_task_during_compaction(
        self,
    ) -> None:
        """All instruction roles keep dynamic context and task during compaction."""
        skill_body = "EXACT_LOADED_SKILL_BODY_71c9"
        memory_fact = "dynamic-memory-fact-8f42"
        task = "portable role task"
        huge_invalid_reply = "x" * 20_000
        budget = 16_000

        for role in ("system", "developer", "user"):
            with self.subTest(role=role):
                role_root = self.root / role
                skill = _rc_skills.Skill(
                    "portable-skill",
                    "Portable skill",
                    skill_body,
                    role_root / "SKILL.md",
                )
                memory = _rc_memory.MemoryStore(role_root / "memory.json")
                chat = ScriptedChat(
                    [
                        '{"action":"skill","name":"portable-skill"}',
                        _json_dump({"action": "remember", "content": memory_fact}),
                        huge_invalid_reply,
                        '{"action":"done","message":"portable"}',
                    ],
                )

                result = registered_run(
                    chat,
                    task,
                    role_root,
                    skills=_rc_skills.SkillStore([skill]),
                    memory=memory,
                    auto_approve=True,
                    context_chars=budget,
                    keep_recent_turns=0,
                    instruction_role=role,
                    max_steps=4,
                )

                self.equal(result, "portable")
                self.equal(len(chat.calls), 4)
                initial_roles = [message["role"] for message in chat.calls[0]]
                if role == "user":
                    self.equal(initial_roles, ["user"])
                    self.require(
                        ("--- USER TASK ---\n" + task) in (chat.calls[0][0]["content"]),
                    )
                else:
                    self.equal(initial_roles, [role, "user"])
                    self.equal(chat.calls[0][1], {"role": "user", "content": task})

                # Skill text and newly persisted memory are rebuilt into the
                # configured leading instruction message on the next request.
                self.require((skill_body) not in (chat.calls[0][0]["content"]))
                self.require((skill_body) in (chat.calls[1][0]["content"]))
                self.require((memory_fact) in (chat.calls[2][0]["content"]))
                expected_before_compaction = (
                    ["user", "assistant", "user", "assistant", "user"]
                    if role == "user"
                    else [
                        role,
                        "user",
                        "assistant",
                        "user",
                        "assistant",
                        "user",
                    ]
                )
                self.equal(
                    [message["role"] for message in chat.calls[2]],
                    expected_before_compaction,
                )

                compacted = chat.calls[3]
                self.require((_rc_context.messages_size(compacted)) <= (budget))
                self.require((huge_invalid_reply) not in (_json_dump(compacted)))
                self.equal(compacted[0]["role"], role)
                self.require((skill_body) in (compacted[0]["content"]))
                self.require((memory_fact) in (compacted[0]["content"]))
                if role == "user":
                    self.equal([message["role"] for message in compacted], ["user"])
                    self.require(
                        ("--- USER TASK ---\n" + task) in (compacted[0]["content"]),
                    )
                    digest_anchor = compacted[0]["content"]
                else:
                    self.equal(
                        [message["role"] for message in compacted],
                        [role, "user"],
                    )
                    self.equal(
                        compacted[1]["content"].split(
                            _rc_context.COMPACTION_SEPARATOR,
                            1,
                        )[0],
                        task,
                    )
                    digest_anchor = compacted[1]["content"]
                self.require((_rc_context.COMPACTION_SEPARATOR) in (digest_anchor))
                self.equal(
                    sum(
                        message["content"].count(_rc_context.COMPACTION_PREFIX)
                        for message in compacted
                    ),
                    1,
                )

    def test_memory_context_is_reduced_to_fit_selected_context(self) -> None:
        """Check memory context is reduced to fit selected context."""
        memory = _rc_memory.MemoryStore(self.root / "memory-fit.json")
        for index in range(12):
            memory.add(f"memory-{index:02d}-" + (str(index % 10) * 900))
        all_items = memory.all()
        self.equal(
            len(array_field(json_object(memory.context()), "memory context")),
            len(all_items),
        )
        chat = ScriptedChat(['{"action":"done","message":"fit"}'])
        budget = 16_000

        result = registered_run(
            chat,
            "fit memories",
            self.root / "memory-workspace",
            memory=memory,
            context_chars=budget,
        )

        self.equal(result, "fit")
        self.require((_rc_context.messages_size(chat.calls[0])) <= (budget))
        rendered_memory = chat.calls[0][0]["content"].split(
            "\nPersistent memories: ",
            1,
        )[1]
        decoded: tuple[object, int] = json.JSONDecoder().raw_decode(rendered_memory)
        selected = array_field(decoded[0], "selected memory")
        end = decoded[1]
        rendered_memory = rendered_memory[:end]
        self.require((len(selected)) > (0))
        self.require((len(selected)) < (len(all_items)))
        self.equal(selected, all_items[-len(selected) :])

        actual_floor = copy.deepcopy(chat.calls[0])
        actual_floor[1]["content"] += (
            _rc_context.COMPACTION_SEPARATOR + _rc_context.COMPACTION_PREFIX
        )
        self.require((_rc_context.messages_size(actual_floor)) <= (budget))

        one_more = all_items[-len(selected) - 1 :]
        expanded_memory = _json_dump(
            one_more,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        expanded_floor = copy.deepcopy(chat.calls[0])
        expanded_floor[0]["content"] = expanded_floor[0]["content"].replace(
            rendered_memory,
            expanded_memory,
            1,
        )
        expanded_floor[1]["content"] += (
            _rc_context.COMPACTION_SEPARATOR + _rc_context.COMPACTION_PREFIX
        )
        self.require((_rc_context.messages_size(expanded_floor)) > (budget))

    def test_near_budget_skill_without_compaction_headroom_is_rejected_recoverably(
        self,
    ) -> None:
        """Near budget skill without compaction headroom is rejected recoverably."""
        budget = 32_000
        task = "load a near-budget skill"
        marker = "NEAR_BUDGET_SKILL_BODY_2a67\n"

        for role in ("system", "developer", "user"):
            with self.subTest(role=role):
                role_root = self.root / ("near-budget-" + role)
                source = role_root / "SKILL.md"
                empty_skill = _rc_skills.Skill(
                    "near-budget",
                    "Near-budget skill",
                    "",
                    source,
                )
                probe = ScriptedChat(
                    [
                        '{"action":"skill","name":"near-budget"}',
                        '{"action":"done","message":"probe"}',
                    ],
                )
                registered_run(
                    probe,
                    task,
                    role_root,
                    skills=_rc_skills.SkillStore([empty_skill]),
                    context_chars=budget,
                    instruction_role=role,
                    max_steps=2,
                )

                body_size = _compaction_body_size(probe, role, budget)
                self.require((body_size) >= (len(marker)))
                self.require((body_size) < (budget))
                body = marker + ("s" * (body_size - len(marker)))
                skill = _rc_skills.Skill(
                    "near-budget",
                    "Near-budget skill",
                    body,
                    source,
                )
                chat = ScriptedChat(
                    [
                        '{"action":"skill","name":"near-budget"}',
                        '{"action":"done","message":"recovered"}',
                    ],
                )

                result = registered_run(
                    chat,
                    task,
                    role_root,
                    skills=_rc_skills.SkillStore([skill]),
                    context_chars=budget,
                    instruction_role=role,
                    max_steps=2,
                )

                self.equal(result, "recovered")
                self.equal(len(chat.calls), 2)
                self.require((_rc_context.messages_size(chat.calls[1])) <= (budget))
                self.require((marker) not in (chat.calls[1][0]["content"]))
                feedback = chat.calls[1][-1]["content"]
                self.require(feedback.startswith(SETTINGS.chat.protocol.result_prefix))
                parsed = _decode_object(
                    feedback[len(SETTINGS.chat.protocol.result_prefix) :],
                )
                self.require(not (parsed["ok"]))
                self.require(("cannot fit") in (text_field(parsed["error"], "error")))
                self.require(
                    ("context budget") in (text_field(parsed["error"], "error")),
                )

    def test_skill_loads_once_and_duplicate_reports_already_loaded(self) -> None:
        """Check skill loads once and duplicate reports already loaded."""
        skill = _rc_skills.Skill(
            "testing",
            "Test guidance",
            "# Trusted skill\nRun unittest.",
            self.root / "SKILL.md",
        )
        chat = ScriptedChat(
            [
                '{"action":"skill","name":"testing"}',
                '{"action":"skill","name":"TESTING"}',
                '{"action":"done","message":"done"}',
            ],
        )
        log = io.StringIO()
        registered_run(
            chat,
            "task",
            self.root / "workspace",
            skills=_rc_skills.SkillStore([skill]),
            log=log,
        )

        final_messages = chat.calls[-1]
        self.equal(sum(item["role"] == "system" for item in final_messages), 1)
        loaded = [
            item
            for item in final_messages
            if item["role"] == "system"
            and "Loaded operator-configured skill" in item["content"]
        ]
        self.equal(len(loaded), 1)
        host_results = [
            _decode_object(item["content"][len(SETTINGS.chat.protocol.result_prefix) :])
            for item in final_messages
            if item["content"].startswith(SETTINGS.chat.protocol.result_prefix)
        ]
        self.equal(host_results[0]["already_loaded"], expected=False)
        self.equal(host_results[1]["already_loaded"], expected=True)
        # The durable log records the action/result history. The trusted skill text
        # is rebuilt into the leading API system prompt instead of being appended
        # as another historical system message.
        self.equal(
            sum(
                "Loaded operator-configured skill" in line
                for line in log.getvalue().splitlines()
            ),
            0,
        )

    def test_unknown_skill_and_disabled_memory_errors_are_returned_to_model(
        self,
    ) -> None:
        """Check unknown skill and disabled memory errors are returned to model."""
        chat = ScriptedChat(
            [
                '{"action":"skill","name":"missing"}',
                '{"action":"memories"}',
                '{"action":"remember","content":"fact"}',
                '{"action":"forget","id":"1"}',
                '{"action":"done","message":"done"}',
            ],
        )
        registered_run(chat, "task", self.root, auto_approve=True)
        final = chat.calls[-1]
        results = [
            _decode_object(item["content"][len(SETTINGS.chat.protocol.result_prefix) :])
            for item in final
            if item["content"].startswith(SETTINGS.chat.protocol.result_prefix)
        ]
        self.equal(len(results), 4)
        self.require(all(not result["ok"] for result in results))
        self.require(("Unknown skill") in (text_field(results[0]["error"], "error")))
        self.require(
            all(
                "disabled" in text_field(result["error"], "error")
                for result in results[1:]
            ),
        )

    def test_memory_actions_persist_and_are_visible_on_next_run(self) -> None:
        """Check memory actions persist and are visible on next run."""
        memory = _rc_memory.MemoryStore(self.root / "memory.json")
        first = ScriptedChat(
            [
                '{"action":"remember","content":"durable fact"}',
                '{"action":"memories"}',
                '{"action":"done","message":"saved"}',
            ],
        )
        registered_run(
            first,
            "first",
            self.root / "one",
            memory=memory,
            auto_approve=True,
        )
        self.equal(memory.all(), [{"id": 1, "content": "durable fact"}])
        # Persistent context is rebuilt before every API call, so a newly saved
        # fact is available immediately without duplicating system messages.
        self.require(("durable fact") in (first.calls[1][0]["content"]))
        self.equal(sum(item["role"] == "system" for item in first.calls[1]), 1)
        results = [
            _decode_object(item["content"][len(SETTINGS.chat.protocol.result_prefix) :])
            for item in first.calls[-1]
            if item["content"].startswith(SETTINGS.chat.protocol.result_prefix)
        ]
        self.equal(results[0], {"ok": True, "id": 1})
        self.equal(
            object_field(array_field(results[1]["memories"], "memories")[0], "memory")[
                "id"
            ],
            1,
        )
        self.equal(
            object_field(array_field(results[1]["memories"], "memories")[0], "memory")[
                "content"
            ],
            "durable fact",
        )
        self.require((results[1]["next_cursor"]) is None)
        self.equal(results[1]["total"], 1)

        second = ScriptedChat(['{"action":"done","message":"loaded"}'])
        reloaded = _rc_memory.MemoryStore(self.root / "memory.json")
        registered_run(second, "second", self.root / "two", memory=reloaded)
        self.require(("durable fact") in (second.calls[0][0]["content"]))

    def test_large_memory_store_is_cursor_paged_without_losing_old_content(
        self,
    ) -> None:
        """Check large memory store is cursor paged without losing old content."""
        memory_path = self.root / "large-memory.json"
        items = []
        for memory_id in range(1, _rc_memory.MAX_MEMORY_ITEMS + 1):
            prefix = f"OLD-{memory_id}-"
            items.append(
                {
                    "id": memory_id,
                    "content": prefix
                    + "x" * (_rc_memory.MAX_MEMORY_CHARS - len(prefix)),
                },
            )
        memory_path.write_text(
            _json_dump(
                {
                    "version": 1,
                    "next_id": _rc_memory.MAX_MEMORY_ITEMS + 1,
                    "memories": items,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        memory = _rc_memory.MemoryStore(memory_path)
        chat = ScriptedChat(
            [
                '{"action":"memories"}',
                '{"action":"memories","cursor":"1"}',
                '{"action":"done","message":"paged"}',
            ],
        )

        registered_run(
            chat,
            "inspect old memories",
            self.root / "large-memory-workspace",
            memory=memory,
        )

        first_follow_up = chat.calls[1]
        second_follow_up = chat.calls[2]
        for request in (first_follow_up, second_follow_up):
            self.require(
                (_rc_context.messages_size(request)) <= (SETTINGS.chat.context_chars),
            )
        first_results = [
            _decode_object(
                message["content"][len(SETTINGS.chat.protocol.result_prefix) :],
            )
            for message in first_follow_up
            if message["content"].startswith(SETTINGS.chat.protocol.result_prefix)
        ]
        second_results = [
            _decode_object(
                message["content"][len(SETTINGS.chat.protocol.result_prefix) :],
            )
            for message in second_follow_up
            if message["content"].startswith(SETTINGS.chat.protocol.result_prefix)
        ]
        self.equal(first_results[-1]["memories"], [items[0]])
        self.equal(first_results[-1]["next_cursor"], "1")
        self.equal(second_results[-1]["memories"], [items[1]])
        self.equal(second_results[-1]["next_cursor"], "2")

    def test_escape_heavy_memory_page_survives_default_context_exactly(self) -> None:
        """Check escape heavy memory page survives default context exactly."""
        memory_path = self.root / "escaped-memory.json"
        pattern = '\\"\n\r\t'
        content = pattern * (_rc_memory.MAX_MEMORY_CHARS // len(pattern))
        self.equal(len(content), _rc_memory.MAX_MEMORY_CHARS)
        memory_path.write_text(
            _json_dump(
                {
                    "version": 1,
                    "next_id": 2,
                    "memories": [{"id": 1, "content": content}],
                },
            ),
            encoding="utf-8",
        )
        memory = _rc_memory.MemoryStore(memory_path)
        chat = ScriptedChat(
            [
                '{"action":"memories"}',
                '{"action":"done","message":"retrieved"}',
            ],
        )

        registered_run(
            chat,
            "retrieve it",
            self.root / "escaped-workspace",
            memory=memory,
        )

        request = chat.calls[1]
        self.require(
            (_rc_context.messages_size(request)) <= (SETTINGS.chat.context_chars),
        )
        results = [
            _decode_object(
                message["content"][len(SETTINGS.chat.protocol.result_prefix) :],
            )
            for message in request
            if message["content"].startswith(SETTINGS.chat.protocol.result_prefix)
        ]
        self.equal(results[-1]["memories"], [{"id": 1, "content": content}])
        self.require((results[-1]["next_cursor"]) is None)

    def test_denied_memory_mutation_is_not_applied(self) -> None:
        """Check denied memory mutation is not applied."""
        memory = _rc_memory.MemoryStore(self.root / "memory.json")
        chat = ScriptedChat(
            [
                '{"action":"remember","content":"do not save"}',
                '{"action":"done","message":"done"}',
            ],
        )
        approve = mock.Mock(return_value=False)
        registered_run(
            chat,
            "task",
            self.root,
            memory=memory,
            approval_callback=approve,
        )
        approve.assert_called_once()
        self.equal(memory.all(), [])
        result = _decode_object(
            chat.calls[-1][-1]["content"][len(SETTINGS.chat.protocol.result_prefix) :],
        )
        self.require(result["denied"])

    def test_forgotten_memory_is_absent_from_the_very_next_api_request(self) -> None:
        """Check forgotten memory is absent from the very next api request."""
        memory = _rc_memory.MemoryStore(self.root / "memory.json")
        item = memory.add("forgettable-value-93f0")
        chat = ScriptedChat(
            [
                _json_dump({"action": "forget", "id": str(item["id"])}),
                '{"action":"done","message":"done"}',
            ],
        )

        registered_run(chat, "task", self.root, memory=memory, auto_approve=True)

        self.require(("forgettable-value-93f0") in (chat.calls[0][0]["content"]))
        self.require(("forgettable-value-93f0") not in (chat.calls[1][0]["content"]))
        self.equal(memory.all(), [])

    def test_auto_approved_write_executes_without_prompt(self) -> None:
        """Check auto approved write executes without prompt."""
        chat = ScriptedChat(
            [
                '{"action":"write","path":"created.txt","content":"hello"}',
                '{"action":"done","message":"done"}',
            ],
        )
        with mock.patch("builtins.input") as approve:
            registered_run(chat, "task", self.root, auto_approve=True)
        approve.assert_not_called()
        self.equal((self.root / "created.txt").read_text(encoding="utf-8"), "hello")

    def test_invalid_reply_is_returned_as_host_result_then_model_can_recover(
        self,
    ) -> None:
        """Check invalid reply is returned as host result then model can recover."""
        chat = ScriptedChat(["not json", '{"action":"done","message":"recovered"}'])
        result = registered_run(chat, "task", self.root)
        self.equal(result, "recovered")
        feedback = chat.calls[1][-1]["content"]
        self.require(feedback.startswith(SETTINGS.chat.protocol.result_prefix))
        parsed = _decode_object(feedback[len(SETTINGS.chat.protocol.result_prefix) :])
        self.require(not (parsed["ok"]))
        self.require(("JSONDecodeError") in (text_field(parsed["error"], "error")))

    def test_full_log_retains_history_while_api_input_is_compacted(self) -> None:
        """Check full log retains history while api input is compacted."""
        huge_invalid_reply = "x" * 20_000
        chat = ScriptedChat([huge_invalid_reply, '{"action":"done","message":"done"}'])
        log = io.StringIO()
        budget = 16_000

        registered_run(
            chat,
            "task",
            self.root,
            log=log,
            context_chars=budget,
            keep_recent_turns=0,
        )

        self.equal(len(chat.calls), 2)
        self.require((_rc_context.messages_size(chat.calls[1])) <= (budget))
        self.equal(chat.calls[1][0], chat.calls[0][0])
        original_task, separator, digest = chat.calls[1][1]["content"].rpartition(
            _rc_context.COMPACTION_SEPARATOR,
        )
        self.equal(original_task, "task")
        self.equal(separator, _rc_context.COMPACTION_SEPARATOR)
        self.require(digest.startswith(_rc_context.COMPACTION_PREFIX))
        self.equal(
            sum(
                item["content"].count(_rc_context.COMPACTION_PREFIX)
                for item in chat.calls[1]
            ),
            1,
        )
        self.equal([item["role"] for item in chat.calls[1]], ["system", "user"])
        self.require((huge_invalid_reply) in (log.getvalue()))


class RunCallbackTests(_RunFixture):
    """Check composed session callback isolation, validation and cancellation."""

    def test_event_callback_routes_request_result_and_done_without_printing(
        self,
    ) -> None:
        """Check event callback routes request result and done without printing."""
        (self.root / "visible.txt").write_text("visible", encoding="utf-8")
        chat = ScriptedChat(
            [
                '{"action":"list","path":"."}',
                '{"action":"done","message":"finished"}',
            ],
        )
        events: list[tuple[str, Mapping[str, object]]] = []

        def on_event(name: str, payload: Mapping[str, object]) -> None:
            events.append((name, copy.deepcopy(payload)))
            # Callback values are detached: a UI may annotate or otherwise
            # mutate them without changing the action or host result.
            if name == "request":
                object_field(payload["action"], "action")["path"] = "missing"
            elif name == "result":
                object_field(payload["result"], "result")["entries"] = []

        with mock.patch("builtins.print") as output:
            result = registered_run(
                chat,
                "inspect",
                self.root,
                event_callback=on_event,
            )

        self.equal(result, "finished")
        output.assert_not_called()
        self.equal(
            [name for name, _payload in events],
            ["request", "result", "request", "done"],
        )
        self.equal(
            events[0][1],
            {
                "step": 1,
                "max_steps": None,
                "action": {"action": "list", "path": "."},
            },
        )
        self.require(
            ("visible.txt")
            in (array_field(_result(events[1][1])["entries"], "entries")),
        )
        self.equal(events[1][1]["action"], events[0][1]["action"])
        self.equal(events[-1][1], {"step": 2, "max_steps": None, "message": "finished"})
        host_result = _decode_object(
            chat.calls[1][-1]["content"][len(SETTINGS.chat.protocol.result_prefix) :],
        )
        self.require(
            ("visible.txt") in (array_field(host_result["entries"], "entries")),
        )

    def test_session_without_ui_callback_does_not_write_to_stdout(self) -> None:
        """Check session without ui callback does not write to stdout."""
        chat = ScriptedChat(
            ['{"action":"list","path":"."}', '{"action":"done","message":"finished"}'],
        )
        with mock.patch("builtins.print") as output:
            result = registered_run(chat, "inspect", self.root)
        self.equal(result, "finished")
        output.assert_not_called()

    def test_event_callback_reports_invalid_reply_as_result_without_action(
        self,
    ) -> None:
        """Check event callback reports invalid reply as result without action."""
        chat = ScriptedChat(["invalid", '{"action":"done","message":"recovered"}'])
        events: list[tuple[str, Mapping[str, object]]] = []

        result = registered_run(
            chat,
            "recover",
            self.root,
            event_callback=_record_events(events),
        )

        self.equal(result, "recovered")
        self.equal([name for name, _payload in events], ["result", "request", "done"])
        self.require((events[0][1]["action"]) is None)
        self.require(not (_result(events[0][1])["ok"]))
        self.require(
            ("JSONDecodeError")
            in (text_field(_result(events[0][1])["error"], "error")),
        )

    def test_approval_callback_supplies_decisions_and_receives_detached_actions(
        self,
    ) -> None:
        """Check approval callback supplies decisions and receives detached actions."""
        chat = ScriptedChat(
            [
                _json_dump({"action": "write", "path": "denied.txt", "content": "no"}),
                _json_dump(
                    {"action": "write", "path": "approved.txt", "content": "yes"},
                ),
                '{"action":"done","message":"complete"}',
            ],
        )
        approval_actions = []
        events: list[tuple[str, Mapping[str, object]]] = []

        approved_action_number = 2

        def decide(action: Action) -> bool:
            approval_actions.append(copy.deepcopy(action))
            action["path"] = "callback-mutated.txt"
            return len(approval_actions) == approved_action_number

        with mock.patch("builtins.input") as interactive:
            result = registered_run(
                chat,
                "write files",
                self.root,
                approval_callback=decide,
                event_callback=_record_events(events),
            )

        self.equal(result, "complete")
        interactive.assert_not_called()
        self.equal(
            [action["path"] for action in approval_actions],
            ["denied.txt", "approved.txt"],
        )
        self.require(not ((self.root / "denied.txt").exists()))
        self.equal((self.root / "approved.txt").read_text(encoding="utf-8"), "yes")
        self.require(not ((self.root / "callback-mutated.txt").exists()))
        results = [_result(payload) for name, payload in events if name == "result"]
        self.require(results[0]["denied"])
        self.require(results[1]["ok"])

    def test_auto_approve_bypasses_both_approval_paths(self) -> None:
        """Check auto approve bypasses both approval paths."""
        chat = ScriptedChat(
            [
                '{"action":"write","path":"auto.txt","content":"yes"}',
                '{"action":"done","message":"complete"}',
            ],
        )
        callback = mock.Mock(side_effect=AssertionError("must not be called"))

        with mock.patch("builtins.input") as interactive:
            result = registered_run(
                chat,
                "write",
                self.root,
                auto_approve=True,
                approval_callback=callback,
            )

        self.equal(result, "complete")
        interactive.assert_not_called()
        callback.assert_not_called()
        self.equal((self.root / "auto.txt").read_text(encoding="utf-8"), "yes")

    def test_frontend_callback_exceptions_propagate(self) -> None:
        """Check frontend callback exceptions propagate."""

        class FrontendError(Exception):
            pass

        for failing_event in ("request", "result", "done"):
            with self.subTest(event=failing_event):
                replies = (
                    ['{"action":"done","message":"complete"}']
                    if failing_event == "done"
                    else [
                        '{"action":"list","path":"."}',
                        '{"action":"done","message":"unreached"}',
                    ]
                )

                def on_event(
                    name: str,
                    _payload: Mapping[str, object],
                    *,
                    expected_event: str = failing_event,
                ) -> None:
                    if name == expected_event:
                        raise FrontendError(name)

                with self.rejected(FrontendError, failing_event):
                    registered_run(
                        ScriptedChat(replies),
                        "task",
                        self.root / failing_event,
                        event_callback=on_event,
                    )

        approval = mock.Mock(side_effect=FrontendError("approval"))
        chat = ScriptedChat(['{"action":"write","path":"never.txt","content":"no"}'])
        with self.rejected(FrontendError, "approval"):
            registered_run(
                chat,
                "task",
                self.root / "approval",
                approval_callback=approval,
                event_callback=_quiet_event,
            )
        self.require(not ((self.root / "approval" / "never.txt").exists()))

    def test_argument_validation_and_step_limit(self) -> None:
        """Check argument validation and step limit."""
        invalid = (
            {"task": "", "max_steps": 1, "timeout": 1},
            {"task": "task", "max_steps": 0, "timeout": 1},
            {"task": "task", "max_steps": True, "timeout": 1},
            {"task": "task", "max_steps": 1, "timeout": 0},
            {"task": "task", "max_steps": 1, "timeout": float("nan")},
            {"task": "task", "max_steps": 1, "timeout": float("inf")},
            {"task": "task", "max_steps": 1, "timeout": True},
            {"task": "task", "max_steps": 1, "timeout": 1, "context_chars": 0},
            {"task": "task", "max_steps": 1, "timeout": 1, "context_chars": True},
            {"task": "task", "max_steps": 1, "timeout": 1, "keep_recent_turns": -1},
            {"task": "task", "max_steps": 1, "timeout": 1, "instruction_role": "tool"},
            {"task": "task", "max_steps": 1, "timeout": 1, "protocol": ""},
            {"task": "task", "max_steps": 1, "timeout": 1, "protocol": 7},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                parameters = dict(arguments)
                task = parameters.pop("task")
                if not isinstance(task, str):
                    self.fail("Malformed run fixture task must be text.")
                with self.rejected(ValueError):
                    registered_run(lambda _messages: "", task, self.root, **parameters)

        chat = ScriptedChat(['{"action":"list","path":"."}'])
        with self.rejected(RuntimeError, "without a done action"):
            registered_run(chat, "task", self.root, max_steps=1)

    def test_run_agent_forwards_cancellation(self) -> None:
        """Check run agent forwards cancellation."""

        class CancelledError(Exception):
            pass

        cancellation = CancelledError("stop")
        chat = mock.Mock()

        def cancel() -> None:
            raise cancellation

        caught = captured(
            CancelledError,
            lambda: registered_run(
                chat,
                "task",
                self.root,
                cancel_check=cancel,
                event_callback=_quiet_event,
            ),
        )
        self.require(caught is cancellation)
        chat.assert_not_called()
