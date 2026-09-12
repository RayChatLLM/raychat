"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest import mock

import raychat._common as _rc__common
from raychat.configuration import SETTINGS
from raychat.sdk import Action
from raychat.type_support import override
from raychat.validation import text_field
from tests.plugin_support import ScriptedChat, plugin_module, registered_run

_rc_context = plugin_module("context")
_rc_memory = plugin_module("memory")
_rc_skills = plugin_module("skills")


class RunAgentTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    @override
    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_done_returns_message_and_initial_context_has_catalog_and_memory(
        self,
    ) -> None:
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

        self.assertEqual(result, "complete")
        self.assertEqual(len(chat.calls), 1)
        system = chat.calls[0][0]["content"]
        self.assertIn('"name": "test"', system)
        self.assertIn("Use UTF-8", system)
        self.assertEqual(chat.calls[0][1], {"role": "user", "content": "build it"})

    def test_protocol_override_is_per_run_and_default_remains_exact(self) -> None:
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

        self.assertEqual(len(custom_chat.calls), 2)
        for call in custom_chat.calls:
            self.assertTrue(call[0]["content"].startswith(custom))
            self.assertNotIn(
                text_field(
                    SETTINGS.plugins.settings["context"]["protocol"], "context.protocol"
                ),
                call[0]["content"],
            )
            self.assertIn(
                plugin_module("filesystem").__plugin_manifest__.instructions,
                call[0]["content"],
            )
        self.assertTrue(
            default_chat.calls[0][0]["content"].startswith(
                text_field(
                    SETTINGS.plugins.settings["context"]["protocol"], "context.protocol"
                ),
            ),
        )
        self.assertIn(
            plugin_module("filesystem").__plugin_manifest__.instructions,
            default_chat.calls[0][0]["content"],
        )

    def test_all_instruction_roles_keep_dynamic_context_and_task_during_compaction(
        self,
    ) -> None:
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
                        json.dumps({"action": "remember", "content": memory_fact}),
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

                self.assertEqual(result, "portable")
                self.assertEqual(len(chat.calls), 4)
                initial_roles = [message["role"] for message in chat.calls[0]]
                if role == "user":
                    self.assertEqual(initial_roles, ["user"])
                    self.assertIn(
                        "--- USER TASK ---\n" + task,
                        chat.calls[0][0]["content"],
                    )
                else:
                    self.assertEqual(initial_roles, [role, "user"])
                    self.assertEqual(
                        chat.calls[0][1],
                        {"role": "user", "content": task},
                    )

                # Skill text and newly persisted memory are rebuilt into the
                # configured leading instruction message on the next request.
                self.assertNotIn(skill_body, chat.calls[0][0]["content"])
                self.assertIn(skill_body, chat.calls[1][0]["content"])
                self.assertIn(memory_fact, chat.calls[2][0]["content"])
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
                self.assertEqual(
                    [message["role"] for message in chat.calls[2]],
                    expected_before_compaction,
                )

                compacted = chat.calls[3]
                self.assertLessEqual(_rc_context.messages_size(compacted), budget)
                self.assertNotIn(huge_invalid_reply, json.dumps(compacted))
                self.assertEqual(compacted[0]["role"], role)
                self.assertIn(skill_body, compacted[0]["content"])
                self.assertIn(memory_fact, compacted[0]["content"])
                if role == "user":
                    self.assertEqual(
                        [message["role"] for message in compacted],
                        ["user"],
                    )
                    self.assertIn("--- USER TASK ---\n" + task, compacted[0]["content"])
                    digest_anchor = compacted[0]["content"]
                else:
                    self.assertEqual(
                        [message["role"] for message in compacted],
                        [role, "user"],
                    )
                    self.assertEqual(
                        compacted[1]["content"].split(
                            _rc_context.COMPACTION_SEPARATOR,
                            1,
                        )[0],
                        task,
                    )
                    digest_anchor = compacted[1]["content"]
                self.assertIn(_rc_context.COMPACTION_SEPARATOR, digest_anchor)
                self.assertEqual(
                    sum(
                        message["content"].count(_rc_context.COMPACTION_PREFIX)
                        for message in compacted
                    ),
                    1,
                )

    def test_memory_context_is_reduced_to_fit_selected_context(self) -> None:
        memory = _rc_memory.MemoryStore(self.root / "memory-fit.json")
        for index in range(12):
            memory.add(f"memory-{index:02d}-" + (str(index % 10) * 900))
        all_items = memory.all()
        self.assertEqual(len(json.loads(memory.context())), len(all_items))
        chat = ScriptedChat(['{"action":"done","message":"fit"}'])
        budget = 16_000

        result = registered_run(
            chat,
            "fit memories",
            self.root / "memory-workspace",
            memory=memory,
            context_chars=budget,
        )

        self.assertEqual(result, "fit")
        self.assertLessEqual(_rc_context.messages_size(chat.calls[0]), budget)
        rendered_memory = chat.calls[0][0]["content"].split(
            "\nPersistent memories: ",
            1,
        )[1]
        selected, end = json.JSONDecoder().raw_decode(rendered_memory)
        rendered_memory = rendered_memory[:end]
        self.assertGreater(len(selected), 0)
        self.assertLess(len(selected), len(all_items))
        self.assertEqual(selected, all_items[-len(selected) :])

        actual_floor = copy.deepcopy(chat.calls[0])
        actual_floor[1]["content"] += (
            _rc_context.COMPACTION_SEPARATOR + _rc_context.COMPACTION_PREFIX
        )
        self.assertLessEqual(_rc_context.messages_size(actual_floor), budget)

        one_more = all_items[-len(selected) - 1 :]
        expanded_memory = json.dumps(
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
        self.assertGreater(_rc_context.messages_size(expanded_floor), budget)

    def test_near_budget_skill_without_compaction_headroom_is_rejected_recoverably(
        self,
    ) -> None:
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

                pinned_count = 1 if role == "user" else 2
                bare_pinned = copy.deepcopy(probe.calls[1][:pinned_count])
                marked_pinned = copy.deepcopy(bare_pinned)
                anchor_index = 0 if role == "user" else 1
                marked_pinned[anchor_index]["content"] += (
                    _rc_context.COMPACTION_SEPARATOR + _rc_context.COMPACTION_PREFIX
                )
                headroom = _rc_context.messages_size(
                    marked_pinned,
                ) - _rc_context.messages_size(bare_pinned)
                body_size = (
                    budget - _rc_context.messages_size(bare_pinned) - headroom + 1
                )
                self.assertGreaterEqual(body_size, len(marker))
                self.assertLess(body_size, 32_000)
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

                self.assertEqual(result, "recovered")
                self.assertEqual(len(chat.calls), 2)
                self.assertLessEqual(_rc_context.messages_size(chat.calls[1]), budget)
                self.assertNotIn(marker, chat.calls[1][0]["content"])
                feedback = chat.calls[1][-1]["content"]
                self.assertTrue(feedback.startswith(_rc__common.RESULT_PREFIX))
                parsed = json.loads(feedback[len(_rc__common.RESULT_PREFIX) :])
                self.assertFalse(parsed["ok"])
                self.assertIn("cannot fit", parsed["error"])
                self.assertIn("context budget", parsed["error"])

    def test_skill_loads_once_and_duplicate_reports_already_loaded(self) -> None:
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
        self.assertEqual(sum(item["role"] == "system" for item in final_messages), 1)
        loaded = [
            item
            for item in final_messages
            if item["role"] == "system"
            and "Loaded operator-configured skill" in item["content"]
        ]
        self.assertEqual(len(loaded), 1)
        host_results = [
            json.loads(item["content"][len(_rc__common.RESULT_PREFIX) :])
            for item in final_messages
            if item["content"].startswith(_rc__common.RESULT_PREFIX)
        ]
        self.assertEqual(host_results[0]["already_loaded"], False)
        self.assertEqual(host_results[1]["already_loaded"], True)
        # The durable log records the action/result history. The trusted skill text
        # is rebuilt into the leading API system prompt instead of being appended
        # as another historical system message.
        self.assertEqual(
            sum(
                "Loaded operator-configured skill" in line
                for line in log.getvalue().splitlines()
            ),
            0,
        )

    def test_unknown_skill_and_disabled_memory_errors_are_returned_to_model(
        self,
    ) -> None:
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
            json.loads(item["content"][len(_rc__common.RESULT_PREFIX) :])
            for item in final
            if item["content"].startswith(_rc__common.RESULT_PREFIX)
        ]
        self.assertEqual(len(results), 4)
        self.assertTrue(all(not result["ok"] for result in results))
        self.assertIn("Unknown skill", results[0]["error"])
        self.assertTrue(all("disabled" in result["error"] for result in results[1:]))

    def test_memory_actions_persist_and_are_visible_on_next_run(self) -> None:
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
        self.assertEqual(memory.all(), [{"id": 1, "content": "durable fact"}])
        # Persistent context is rebuilt before every API call, so a newly saved
        # fact is available immediately without duplicating system messages.
        self.assertIn("durable fact", first.calls[1][0]["content"])
        self.assertEqual(sum(item["role"] == "system" for item in first.calls[1]), 1)
        results = [
            json.loads(item["content"][len(_rc__common.RESULT_PREFIX) :])
            for item in first.calls[-1]
            if item["content"].startswith(_rc__common.RESULT_PREFIX)
        ]
        self.assertEqual(results[0], {"ok": True, "id": 1})
        self.assertEqual(results[1]["memories"][0]["id"], 1)
        self.assertEqual(results[1]["memories"][0]["content"], "durable fact")
        self.assertIsNone(results[1]["next_cursor"])
        self.assertEqual(results[1]["total"], 1)

        second = ScriptedChat(['{"action":"done","message":"loaded"}'])
        reloaded = _rc_memory.MemoryStore(self.root / "memory.json")
        registered_run(second, "second", self.root / "two", memory=reloaded)
        self.assertIn("durable fact", second.calls[0][0]["content"])

    def test_large_memory_store_is_cursor_paged_without_losing_old_content(
        self,
    ) -> None:
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
            json.dumps(
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
            self.assertLessEqual(
                _rc_context.messages_size(request),
                _rc__common.DEFAULT_CONTEXT_CHARS,
            )
        first_results = [
            json.loads(message["content"][len(_rc__common.RESULT_PREFIX) :])
            for message in first_follow_up
            if message["content"].startswith(_rc__common.RESULT_PREFIX)
        ]
        second_results = [
            json.loads(message["content"][len(_rc__common.RESULT_PREFIX) :])
            for message in second_follow_up
            if message["content"].startswith(_rc__common.RESULT_PREFIX)
        ]
        self.assertEqual(first_results[-1]["memories"], [items[0]])
        self.assertEqual(first_results[-1]["next_cursor"], "1")
        self.assertEqual(second_results[-1]["memories"], [items[1]])
        self.assertEqual(second_results[-1]["next_cursor"], "2")

    def test_escape_heavy_memory_page_survives_default_context_exactly(self) -> None:
        memory_path = self.root / "escaped-memory.json"
        pattern = '\\"\n\r\t'
        content = pattern * (_rc_memory.MAX_MEMORY_CHARS // len(pattern))
        self.assertEqual(len(content), _rc_memory.MAX_MEMORY_CHARS)
        memory_path.write_text(
            json.dumps(
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
        self.assertLessEqual(
            _rc_context.messages_size(request),
            _rc__common.DEFAULT_CONTEXT_CHARS,
        )
        results = [
            json.loads(message["content"][len(_rc__common.RESULT_PREFIX) :])
            for message in request
            if message["content"].startswith(_rc__common.RESULT_PREFIX)
        ]
        self.assertEqual(
            results[-1]["memories"],
            [{"id": 1, "content": content}],
        )
        self.assertIsNone(results[-1]["next_cursor"])

    def test_denied_memory_mutation_is_not_applied(self) -> None:
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
        self.assertEqual(memory.all(), [])
        result = json.loads(
            chat.calls[-1][-1]["content"][len(_rc__common.RESULT_PREFIX) :],
        )
        self.assertTrue(result["denied"])

    def test_forgotten_memory_is_absent_from_the_very_next_api_request(self) -> None:
        memory = _rc_memory.MemoryStore(self.root / "memory.json")
        item = memory.add("forgettable-value-93f0")
        chat = ScriptedChat(
            [
                json.dumps({"action": "forget", "id": str(item["id"])}),
                '{"action":"done","message":"done"}',
            ],
        )

        registered_run(chat, "task", self.root, memory=memory, auto_approve=True)

        self.assertIn("forgettable-value-93f0", chat.calls[0][0]["content"])
        self.assertNotIn("forgettable-value-93f0", chat.calls[1][0]["content"])
        self.assertEqual(memory.all(), [])

    def test_auto_approved_write_executes_without_prompt(self) -> None:
        chat = ScriptedChat(
            [
                '{"action":"write","path":"created.txt","content":"hello"}',
                '{"action":"done","message":"done"}',
            ],
        )
        with mock.patch("builtins.input") as approve:
            registered_run(chat, "task", self.root, auto_approve=True)
        approve.assert_not_called()
        self.assertEqual(
            (self.root / "created.txt").read_text(encoding="utf-8"),
            "hello",
        )

    def test_invalid_reply_is_returned_as_host_result_then_model_can_recover(
        self,
    ) -> None:
        chat = ScriptedChat(["not json", '{"action":"done","message":"recovered"}'])
        result = registered_run(chat, "task", self.root)
        self.assertEqual(result, "recovered")
        feedback = chat.calls[1][-1]["content"]
        self.assertTrue(feedback.startswith(_rc__common.RESULT_PREFIX))
        parsed = json.loads(feedback[len(_rc__common.RESULT_PREFIX) :])
        self.assertFalse(parsed["ok"])
        self.assertIn("JSONDecodeError", parsed["error"])

    def test_full_log_retains_history_while_api_input_is_compacted(self) -> None:
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

        self.assertEqual(len(chat.calls), 2)
        self.assertLessEqual(_rc_context.messages_size(chat.calls[1]), budget)
        self.assertEqual(chat.calls[1][0], chat.calls[0][0])
        original_task, separator, digest = chat.calls[1][1]["content"].rpartition(
            _rc_context.COMPACTION_SEPARATOR,
        )
        self.assertEqual(original_task, "task")
        self.assertEqual(separator, _rc_context.COMPACTION_SEPARATOR)
        self.assertTrue(digest.startswith(_rc_context.COMPACTION_PREFIX))
        self.assertEqual(
            sum(
                item["content"].count(_rc_context.COMPACTION_PREFIX)
                for item in chat.calls[1]
            ),
            1,
        )
        self.assertEqual([item["role"] for item in chat.calls[1]], ["system", "user"])
        self.assertIn(huge_invalid_reply, log.getvalue())

    def test_event_callback_routes_request_result_and_done_without_printing(
        self,
    ) -> None:
        (self.root / "visible.txt").write_text("visible", encoding="utf-8")
        chat = ScriptedChat(
            [
                '{"action":"list","path":"."}',
                '{"action":"done","message":"finished"}',
            ],
        )
        events = []

        def on_event(name: str, payload: Mapping[str, Any]) -> None:
            events.append((name, copy.deepcopy(payload)))
            # Callback values are detached: a UI may annotate or otherwise
            # mutate them without changing the action or host result.
            if name == "request":
                payload["action"]["path"] = "missing"
            elif name == "result":
                payload["result"]["entries"] = []

        with mock.patch("builtins.print") as output:
            result = registered_run(
                chat,
                "inspect",
                self.root,
                event_callback=on_event,
            )

        self.assertEqual(result, "finished")
        output.assert_not_called()
        self.assertEqual(
            [name for name, _payload in events],
            ["request", "result", "request", "done"],
        )
        self.assertEqual(
            events[0][1],
            {
                "step": 1,
                "max_steps": None,
                "action": {"action": "list", "path": "."},
            },
        )
        self.assertIn("visible.txt", events[1][1]["result"]["entries"])
        self.assertEqual(events[1][1]["action"], events[0][1]["action"])
        self.assertEqual(
            events[-1][1],
            {"step": 2, "max_steps": None, "message": "finished"},
        )
        host_result = json.loads(
            chat.calls[1][-1]["content"][len(_rc__common.RESULT_PREFIX) :],
        )
        self.assertIn("visible.txt", host_result["entries"])

    def test_session_without_ui_callback_does_not_write_to_stdout(self) -> None:
        chat = ScriptedChat(
            ['{"action":"list","path":"."}', '{"action":"done","message":"finished"}'],
        )
        with mock.patch("builtins.print") as output:
            result = registered_run(chat, "inspect", self.root)
        self.assertEqual(result, "finished")
        output.assert_not_called()

    def test_event_callback_reports_invalid_reply_as_result_without_action(
        self,
    ) -> None:
        chat = ScriptedChat(["invalid", '{"action":"done","message":"recovered"}'])
        events = []

        result = registered_run(
            chat,
            "recover",
            self.root,
            event_callback=lambda name, payload: events.append((name, payload)),
        )

        self.assertEqual(result, "recovered")
        self.assertEqual(
            [name for name, _payload in events],
            ["result", "request", "done"],
        )
        self.assertIsNone(events[0][1]["action"])
        self.assertFalse(events[0][1]["result"]["ok"])
        self.assertIn("JSONDecodeError", events[0][1]["result"]["error"])

    def test_approval_callback_supplies_decisions_and_receives_detached_actions(
        self,
    ) -> None:
        chat = ScriptedChat(
            [
                json.dumps({"action": "write", "path": "denied.txt", "content": "no"}),
                json.dumps(
                    {"action": "write", "path": "approved.txt", "content": "yes"},
                ),
                '{"action":"done","message":"complete"}',
            ],
        )
        approval_actions = []
        events = []

        def decide(action: Action) -> bool:
            approval_actions.append(copy.deepcopy(action))
            action["path"] = "callback-mutated.txt"
            return len(approval_actions) == 2

        with mock.patch("builtins.input") as interactive:
            result = registered_run(
                chat,
                "write files",
                self.root,
                approval_callback=decide,
                event_callback=lambda name, payload: events.append((name, payload)),
            )

        self.assertEqual(result, "complete")
        interactive.assert_not_called()
        self.assertEqual(
            [action["path"] for action in approval_actions],
            ["denied.txt", "approved.txt"],
        )
        self.assertFalse((self.root / "denied.txt").exists())
        self.assertEqual(
            (self.root / "approved.txt").read_text(encoding="utf-8"),
            "yes",
        )
        self.assertFalse((self.root / "callback-mutated.txt").exists())
        results = [payload["result"] for name, payload in events if name == "result"]
        self.assertTrue(results[0]["denied"])
        self.assertTrue(results[1]["ok"])

    def test_auto_approve_bypasses_both_approval_paths(self) -> None:
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

        self.assertEqual(result, "complete")
        interactive.assert_not_called()
        callback.assert_not_called()
        self.assertEqual((self.root / "auto.txt").read_text(encoding="utf-8"), "yes")

    def test_frontend_callback_exceptions_propagate(self) -> None:
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
                    _payload: Mapping[str, Any],
                    *,
                    expected_event: str = failing_event,
                ) -> None:
                    if name == expected_event:
                        raise FrontendError(name)

                with self.assertRaisesRegex(FrontendError, failing_event):
                    registered_run(
                        ScriptedChat(replies),
                        "task",
                        self.root / failing_event,
                        event_callback=on_event,
                    )

        approval = mock.Mock(side_effect=FrontendError("approval"))
        chat = ScriptedChat(['{"action":"write","path":"never.txt","content":"no"}'])
        with self.assertRaisesRegex(FrontendError, "approval"):
            registered_run(
                chat,
                "task",
                self.root / "approval",
                approval_callback=approval,
                event_callback=lambda _name, _payload: None,
            )
        self.assertFalse((self.root / "approval" / "never.txt").exists())

    def test_argument_validation_and_step_limit(self) -> None:
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
                task = cast("str", parameters.pop("task"))
                with self.assertRaises(ValueError):
                    registered_run(lambda messages: "", task, self.root, **parameters)

        chat = ScriptedChat(['{"action":"list","path":"."}'])
        with self.assertRaisesRegex(RuntimeError, "without a done action"):
            registered_run(chat, "task", self.root, max_steps=1)

    def test_run_agent_forwards_cancellation(self) -> None:
        class Cancelled(Exception):
            pass

        cancellation = Cancelled("stop")
        chat = mock.Mock()

        def cancel() -> None:
            raise cancellation

        with self.assertRaises(Cancelled) as caught:
            registered_run(
                chat,
                "task",
                self.root,
                cancel_check=cancel,
                event_callback=lambda _name, _payload: None,
            )

        self.assertIs(caught.exception, cancellation)
        chat.assert_not_called()
