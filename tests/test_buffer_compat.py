from __future__ import annotations

from dataclasses import dataclass

from nekro_plugin_debounce.compat import (
    has_hard_boundary,
    text_compatible,
    usage_scope_matches,
)


@dataclass
class Segment:
    type: str
    text: str

    def model_dump(self, **_kwargs):
        return {"type": self.type, "text": self.text}


@dataclass
class Message:
    chat_key: str = "chat"
    chat_type: str = "group"
    content_text: str = ""
    content_data: list[Segment] | list[dict] | None = None


def test_text_and_at_are_classifier_safe() -> None:
    message = Message(content_text="hello", content_data=[Segment("text", "hello"), Segment("at", "@bot")])
    assert text_compatible(message)
    assert not has_hard_boundary(message)


def test_media_is_hard_boundary() -> None:
    current = Message(content_text="图片", content_data=[Segment("image", "[图片]")])
    assert has_hard_boundary(current)


def test_usage_scope_matches_chat_type() -> None:
    group = Message(chat_type="group")
    private = Message(chat_type="private")
    assert usage_scope_matches(group, "both")
    assert usage_scope_matches(group, "group")
    assert not usage_scope_matches(group, "private")
    assert usage_scope_matches(private, "private")
