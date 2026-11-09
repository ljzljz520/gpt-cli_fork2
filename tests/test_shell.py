from unittest import mock

from gptcli.cli import citation_snippet, render_sources
from gptcli.completion import Citation, CitationsEvent, MessageDeltaEvent
from gptcli.shell import simple_response


def make_assistant(events):
    assistant = mock.MagicMock()
    assistant.init_messages.return_value = []
    assistant.complete_chat.return_value = iter(events)
    return assistant


def _events():
    return [
        MessageDeltaEvent("中文回答"),
        MessageDeltaEvent(" English"),
        CitationsEvent(
            [
                Citation(
                    url="https://example.com/source",
                    title="Source Title",
                    start_index=0,
                    end_index=4,
                )
            ]
        ),
    ]


def test_piped_output_is_pure_body_by_default(capsys):
    assistant = make_assistant(_events())
    simple_response(assistant, "question", stream=True)

    captured = capsys.readouterr()
    assert captured.out == "中文回答 English"
    assert "Sources" not in captured.out
    assert "https://" not in captured.out


def test_piped_output_appends_sources_only_when_requested(capsys):
    assistant = make_assistant(_events())
    simple_response(assistant, "question", stream=True, show_sources=True)

    captured = capsys.readouterr()
    assert captured.out.startswith("中文回答 English")
    assert "Sources:" in captured.out
    assert "https://example.com/source" in captured.out
    assert "Source Title" in captured.out
    # The cited Chinese span is located via character offsets
    assert "中文回答" in captured.out


def test_non_stream_piped_output_is_pure_body_by_default(capsys):
    assistant = make_assistant(_events())
    simple_response(assistant, "question", stream=False, show_sources=False)

    captured = capsys.readouterr()
    assert captured.out == "中文回答 English"


def test_citation_snippet_handles_chinese_offsets_and_clamps():
    text = "中文回答内容"
    assert citation_snippet(text, 0, 2) == "中文"
    # Out-of-range indices are clamped instead of raising
    assert citation_snippet(text, 100, 200) == ""
    assert citation_snippet(text, -5, 2) == "中文"


def test_render_sources_never_rewrites_body():
    text = "plain body"
    citations = [
        Citation(url="https://example.com", title="T", start_index=0, end_index=5)
    ]
    rendered = render_sources(citations, text, markdown=False)
    assert text not in rendered  # body is not embedded/rewritten, only snippets
    assert rendered.startswith("Sources:")
    assert "https://example.com" in rendered

    markdown_rendered = render_sources(citations, text, markdown=True)
    assert "[T](https://example.com)" in markdown_rendered


def test_render_sources_dedupes_identical_citations():
    citations = [
        Citation(url="https://example.com", title="T", start_index=0, end_index=1),
        Citation(url="https://example.com", title="T", start_index=0, end_index=1),
    ]
    rendered = render_sources(citations, "ab", markdown=False)
    assert rendered.count("[1]") == 1
    assert "[2]" not in rendered
