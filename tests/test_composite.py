from unittest import mock

from gptcli.completion import Citation, UsageEvent
from gptcli.composite import CompositeChatListener, CompositeResponseStreamer
from gptcli.cost import PriceChatListener
from gptcli.session import ChatListener, ResponseStreamer


def test_composite_streamer_forwards_citations():
    streamer_a = mock.MagicMock(spec=ResponseStreamer)
    streamer_b = mock.MagicMock(spec=ResponseStreamer)
    composite = CompositeResponseStreamer([streamer_a, streamer_b])

    citations = [Citation(url="https://example.com", start_index=0, end_index=1)]
    composite.on_citations(citations)

    streamer_a.on_citations.assert_called_once_with(citations)
    streamer_b.on_citations.assert_called_once_with(citations)


def test_composite_listener_forwards_citations_to_all_listeners():
    plain_listener = mock.MagicMock(spec=ChatListener)
    # The real price listener uses the base signature with the citations
    # keyword argument and must not break when citations are forwarded.
    price_listener = PriceChatListener(mock.MagicMock())

    composite = CompositeChatListener([plain_listener, price_listener])
    usage = UsageEvent(
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
        cost=0.0,
    )
    citations = [Citation(url="https://example.com", start_index=0, end_index=1)]

    composite.on_chat_response(
        [],
        {"role": "assistant", "content": "response"},
        usage,
        citations=citations,
    )

    plain_listener.on_chat_response.assert_called_once_with(
        [],
        {"role": "assistant", "content": "response"},
        usage,
        citations=citations,
    )
