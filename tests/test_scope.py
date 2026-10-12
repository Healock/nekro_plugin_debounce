from __future__ import annotations

from nekro_plugin_debounce.scope import ReplyScopeStore


def test_scope_keeps_same_sender_and_expires() -> None:
    store = ReplyScopeStore(ttl_seconds=10)
    first = store.begin(
        chat_key="group",
        generation=1,
        trigger_source="is_tome",
        message_id="a1",
        sender_id="a",
        sender_name="A",
        platform_userid="a",
        now=100,
    )
    continued = store.begin(
        chat_key="group",
        generation=1,
        trigger_source="same_sender_continuation",
        message_id="a2",
        sender_id="a",
        sender_name="A",
        platform_userid="a",
        now=105,
    )

    assert continued is first
    assert continued.state == "active"
    assert continued.representative_message_id == "a2"
    assert store.get("group", now=114) is continued
    assert store.get("group", now=115) is None


def test_different_trigger_sender_marks_scope_ambiguous() -> None:
    store = ReplyScopeStore()
    store.begin(
        chat_key="group",
        generation=1,
        trigger_source="is_tome",
        message_id="a1",
        sender_id="a",
        sender_name="A",
        platform_userid="a",
        now=100,
    )
    scope = store.begin(
        chat_key="group",
        generation=2,
        trigger_source="preset_name",
        message_id="b1",
        sender_id="b",
        sender_name="B",
        platform_userid="b",
        now=101,
    )

    assert scope.state == "ambiguous"
    rendered = store.render("group")
    assert '"state":"ambiguous"' in rendered
    assert "不要指定、点名或 @ 任何成员" in rendered
    assert '"sender_name":"B"' not in rendered


def test_missing_and_canceled_scope_use_safe_prompt() -> None:
    store = ReplyScopeStore()
    assert '"state":"missing"' in store.render("unknown")
    store.begin(
        chat_key="group",
        generation=1,
        trigger_source="is_tome",
        message_id="a1",
        sender_id="a",
        sender_name="A",
        platform_userid="a",
    )
    store.cancel("group")
    assert '"state":"canceled"' in store.render("group")
    assert "不要指定、点名或 @ 任何成员" in store.render("group")
