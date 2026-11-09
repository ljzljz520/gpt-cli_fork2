from abc import abstractmethod
from typing import Iterator, List, Literal, Optional, TypedDict, Union

from attr import dataclass


class Message(TypedDict):
    role: str
    content: str


@dataclass
class Citation:
    """A web source citation attached to a span of the response text.

    ``start_index`` and ``end_index`` are Python character (Unicode code point)
    offsets into the full, unmodified response text, so slicing works correctly
    for non-ASCII text such as Chinese.
    """

    url: str
    start_index: int
    end_index: int
    title: Optional[str] = None


@dataclass
class CitationsEvent:
    citations: List[Citation]
    type: Literal["citations"] = "citations"


class Pricing(TypedDict):
    prompt: float
    response: float


@dataclass
class MessageDeltaEvent:
    text: str
    type: Literal["message_delta"] = "message_delta"


@dataclass
class ThinkingDeltaEvent:
    text: str
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass
class ToolCallEvent:
    text: str
    type: Literal["tool_call"] = "tool_call"


@dataclass
class UsageEvent:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    type: Literal["usage"] = "usage"

    @staticmethod
    def with_pricing(
        prompt_tokens: int, completion_tokens: int, total_tokens: int, pricing: Pricing
    ) -> "UsageEvent":
        return UsageEvent(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=prompt_tokens * pricing["prompt"]
            + completion_tokens * pricing["response"],
        )


CompletionEvent = Union[
    MessageDeltaEvent, ThinkingDeltaEvent, UsageEvent, ToolCallEvent, CitationsEvent
]


class CompletionProvider:
    @abstractmethod
    def complete(
        self, messages: List[Message], args: dict, stream: bool = False
    ) -> Iterator[CompletionEvent]:
        pass


class CompletionError(Exception):
    pass


class BadRequestError(CompletionError):
    pass
