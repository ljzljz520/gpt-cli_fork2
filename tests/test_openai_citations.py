from types import SimpleNamespace
from unittest import mock

from gptcli.completion import Citation, CitationsEvent, MessageDeltaEvent, UsageEvent
from gptcli.providers.openai import (
    OpenAICompletionProvider,
    extract_url_citations,
)


def url_citation(start_index, end_index, url="https://example.com", title="Example"):
    return SimpleNamespace(
        type="url_citation",
        start_index=start_index,
        end_index=end_index,
        url=url,
        title=title,
    )


def file_citation():
    return SimpleNamespace(type="file_citation", file_id="file-1", index=0)


def text_part(text, annotations=None):
    return SimpleNamespace(type="output_text", text=text, annotations=annotations or [])


def message_item(item_id, parts):
    return SimpleNamespace(type="message", id=item_id, content=parts)


def web_search_item(item_id):
    return SimpleNamespace(type="web_search_call", id=item_id, action=None)


def make_response():
    # Two message output items separated by a non-text item. Annotation offsets
    # are item-local, as returned by the API; the text includes Chinese so the
    # character (not byte) offset handling is exercised.
    item_0 = message_item(
        "msg_1",
        [
            text_part(
                "中文回答内容",
                [
                    url_citation(0, 2, url="https://a.example", title="甲"),
                    file_citation(),
                ],
            )
        ],
    )
    item_1 = message_item(
        "msg_2",
        [
            text_part(
                "English text",
                [url_citation(0, 7, url="https://b.example", title="乙")],
            )
        ],
    )
    response = SimpleNamespace(
        output=[item_0, web_search_item("ws_1"), item_1],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    response.output_text = "中文回答内容English text"
    return response


def make_provider(create_side_effect):
    provider = OpenAICompletionProvider(api_key="test-key")
    provider.client = mock.MagicMock()
    provider.client.responses.create.side_effect = create_side_effect
    return provider


def test_extract_url_citations_offsets_and_filtering():
    response = make_response()
    citations = extract_url_citations(response)

    assert citations == [
        Citation(
            url="https://a.example",
            title="甲",
            start_index=0,
            end_index=2,
        ),
        Citation(
            url="https://b.example",
            title="乙",
            # 6 Chinese characters from the first message item shift the
            # item-local offset (0, 7) into global response-text offsets.
            start_index=6,
            end_index=13,
        ),
    ]

    full_text = response.output_text
    assert full_text[citations[0].start_index : citations[0].end_index] == "中文"
    assert full_text[citations[1].start_index : citations[1].end_index] == "English"


def test_extract_url_citations_dedupes_exact_duplicates():
    response = SimpleNamespace(
        output=[
            message_item(
                "msg_1",
                [
                    text_part(
                        "abc",
                        [
                            url_citation(0, 1, url="https://a.example"),
                            url_citation(0, 1, url="https://a.example"),
                        ],
                    )
                ],
            )
        ]
    )
    citations = extract_url_citations(response)
    assert citations == [
        Citation(url="https://a.example", title="Example", start_index=0, end_index=1)
    ]


def test_stream_and_non_stream_produce_equivalent_citations():
    response = make_response()
    full_text = response.output_text

    def stream_events():
        # Text deltas for the first message item
        for char in "中文回答内容":
            yield SimpleNamespace(
                type="response.output_text.delta",
                delta=char,
                output_index=0,
                content_index=0,
                item_id="msg_1",
            )
        # The citation event arrives *after* the text it annotates
        yield SimpleNamespace(
            type="response.output_text.annotation.added",
            annotation=url_citation(0, 2, url="https://a.example", title="甲"),
            output_index=0,
            content_index=0,
            annotation_index=0,
            item_id="msg_1",
        )
        # Deltas for the second message item (output index 2, index 1 is the
        # web search call item)
        for char in "English text":
            yield SimpleNamespace(
                type="response.output_text.delta",
                delta=char,
                output_index=2,
                content_index=0,
                item_id="msg_2",
            )
        yield SimpleNamespace(
            type="response.output_text.annotation.added",
            annotation=url_citation(0, 7, url="https://b.example", title="乙"),
            output_index=2,
            content_index=0,
            annotation_index=0,
            item_id="msg_2",
        )
        yield SimpleNamespace(type="response.completed", response=response)

    def fake_create(**kwargs):
        if kwargs["stream"]:
            return iter(stream_events())
        return response

    provider = make_provider(fake_create)
    args = {"model": "o3"}

    stream_events_list = list(provider.complete([], args, stream=True))
    non_stream_events_list = list(provider.complete([], args, stream=False))

    def split(events):
        text = "".join(
            event.text for event in events if isinstance(event, MessageDeltaEvent)
        )
        citation_events = [
            event for event in events if isinstance(event, CitationsEvent)
        ]
        usage_events = [event for event in events if isinstance(event, UsageEvent)]
        return text, citation_events, usage_events

    stream_text, stream_citation_events, stream_usage = split(stream_events_list)
    block_text, block_citation_events, block_usage = split(non_stream_events_list)

    # The body text is identical and untouched in both modes
    assert stream_text == block_text == full_text

    # Citations are equivalent between streaming and non-streaming modes
    assert len(stream_citation_events) == len(block_citation_events) == 1
    assert (
        stream_citation_events[0].citations
        == block_citation_events[0].citations
        == extract_url_citations(response)
    )

    # Usage still flows through in both modes
    assert len(stream_usage) == len(block_usage) == 1
    assert stream_usage[0].total_tokens == 18


def test_stream_fallback_uses_late_annotation_events_without_completed():
    # If the terminal response.completed event is missing, annotation.added
    # events must still be surfaced, with global offsets rebuilt from delta
    # lengths (including the output index gap left by the web search item).
    def stream_events():
        for char in "中文回答内容":
            yield SimpleNamespace(
                type="response.output_text.delta",
                delta=char,
                output_index=0,
                content_index=0,
                item_id="msg_1",
            )
        for char in "English text":
            yield SimpleNamespace(
                type="response.output_text.delta",
                delta=char,
                output_index=2,
                content_index=0,
                item_id="msg_2",
            )
        yield SimpleNamespace(
            type="response.output_text.annotation.added",
            annotation=url_citation(0, 7, url="https://b.example", title="乙"),
            output_index=2,
            content_index=0,
            annotation_index=0,
            item_id="msg_2",
        )

    provider = make_provider(lambda **kwargs: iter(stream_events()))
    events = list(provider.complete([], {"model": "o3"}, stream=True))

    citation_events = [event for event in events if isinstance(event, CitationsEvent)]
    assert len(citation_events) == 1
    assert citation_events[0].citations == [
        Citation(url="https://b.example", title="乙", start_index=6, end_index=13)
    ]


def test_no_citations_means_no_citation_event():
    response = SimpleNamespace(
        output=[message_item("msg_1", [text_part("plain answer", [])])],
        usage=None,
    )
    response.output_text = "plain answer"

    provider = make_provider(lambda **kwargs: response)
    events = list(provider.complete([], {"model": "o3"}, stream=False))
    assert not [event for event in events if isinstance(event, CitationsEvent)]
