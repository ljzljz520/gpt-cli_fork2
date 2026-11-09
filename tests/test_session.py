from unittest import mock
from gptcli.completion import (
    BadRequestError,
    Citation,
    CitationsEvent,
    CompletionError,
    MessageDeltaEvent,
)
from gptcli.session import ChatSession

system_message = {"role": "system", "content": "system message"}


def setup_assistant_mock():
    assistant_mock = mock.MagicMock()
    assistant_mock.init_messages.return_value = [system_message]
    return assistant_mock


def setup_listener_mock():
    listener_mock = mock.MagicMock()
    response_streamer_mock = mock.MagicMock()
    response_streamer_mock.__enter__.return_value = response_streamer_mock
    listener_mock.response_streamer.return_value = response_streamer_mock
    return listener_mock, response_streamer_mock


def setup_session():
    assistant_mock = setup_assistant_mock()
    listener_mock, _ = setup_listener_mock()
    session = ChatSession(assistant_mock, listener_mock)
    return assistant_mock, listener_mock, session


def test_simple_input():
    assistant_mock, listener_mock, session = setup_session()

    expected_response = "assistant message"
    assistant_mock.complete_chat.return_value = [MessageDeltaEvent(expected_response)]

    user_input = "user message"
    should_continue = session.process_input(user_input)
    assert should_continue

    user_message = {"role": "user", "content": user_input}
    assistant_message = {"role": "assistant", "content": expected_response}

    assistant_mock.complete_chat.assert_called_once_with(
        [system_message, user_message],
        stream=True,
    )
    listener_mock.on_chat_message.assert_has_calls(
        [mock.call(user_message), mock.call(assistant_message)]
    )


def test_quit():
    _, _, session = setup_session()
    should_continue = session.process_input(":q")
    assert not should_continue


def test_clear():
    assistant_mock, listener_mock, session = setup_session()

    assistant_mock.init_messages.assert_called_once()
    assistant_mock.init_messages.reset_mock()

    assistant_mock.complete_chat.return_value = [MessageDeltaEvent("assistant_message")]

    should_continue = session.process_input("user_message")
    assert should_continue

    assistant_mock.complete_chat.assert_called_once_with(
        [system_message, {"role": "user", "content": "user_message"}],
        stream=True,
    )
    listener_mock.on_chat_message.assert_has_calls(
        [
            mock.call({"role": "user", "content": "user_message"}),
            mock.call({"role": "assistant", "content": "assistant_message"}),
        ]
    )
    assistant_mock.complete_chat.reset_mock()
    listener_mock.on_chat_message.reset_mock()

    should_continue = session.process_input(":c")
    assert should_continue

    assistant_mock.init_messages.assert_called_once()
    listener_mock.on_chat_clear.assert_called_once()
    assistant_mock.complete_chat.assert_not_called()

    assistant_mock.complete_chat.return_value = [
        MessageDeltaEvent("assistant_message_1")
    ]

    should_continue = session.process_input("user_message_1")
    assert should_continue

    assistant_mock.complete_chat.assert_called_once_with(
        [system_message, {"role": "user", "content": "user_message_1"}],
        stream=True,
    )
    listener_mock.on_chat_message.assert_has_calls(
        [
            mock.call({"role": "user", "content": "user_message_1"}),
            mock.call({"role": "assistant", "content": "assistant_message_1"}),
        ]
    )


def test_rerun():
    assistant_mock, listener_mock, session = setup_session()

    assistant_mock.init_messages.assert_called_once()
    assistant_mock.init_messages.reset_mock()

    # Re-run before any input shouldn't do anything
    should_continue = session.process_input(":r")
    assert should_continue

    assistant_mock.init_messages.assert_not_called()
    assistant_mock.complete_chat.assert_not_called()
    listener_mock.on_chat_message.assert_not_called()
    listener_mock.on_chat_rerun.assert_called_once_with(False)

    listener_mock.on_chat_rerun.reset_mock()

    # Now proper re-run
    assistant_mock.complete_chat.return_value = [MessageDeltaEvent("assistant_message")]

    should_continue = session.process_input("user_message")
    assert should_continue

    assistant_mock.complete_chat.assert_called_once_with(
        [system_message, {"role": "user", "content": "user_message"}],
        stream=True,
    )
    listener_mock.on_chat_message.assert_has_calls(
        [
            mock.call({"role": "user", "content": "user_message"}),
            mock.call({"role": "assistant", "content": "assistant_message"}),
        ]
    )
    assistant_mock.complete_chat.reset_mock()
    listener_mock.on_chat_message.reset_mock()

    assistant_mock.complete_chat.return_value = [
        MessageDeltaEvent("assistant_message_1")
    ]

    should_continue = session.process_input(":r")
    assert should_continue

    listener_mock.on_chat_rerun.assert_called_once_with(True)

    assistant_mock.complete_chat.assert_called_once_with(
        [system_message, {"role": "user", "content": "user_message"}],
        stream=True,
    )
    listener_mock.on_chat_message.assert_has_calls(
        [
            mock.call({"role": "assistant", "content": "assistant_message_1"}),
        ]
    )


def test_invalid_request_error():
    assistant_mock, listener_mock, session = setup_session()

    error = BadRequestError("error message")
    assistant_mock.complete_chat.side_effect = error

    user_input = "user message"
    should_continue = session.process_input(user_input)
    assert should_continue

    user_message = {"role": "user", "content": user_input}
    listener_mock.on_chat_message.assert_has_calls([mock.call(user_message)])
    listener_mock.on_error.assert_called_once_with(error)

    # Now rerun shouldn't do anything because user input was not saved
    assistant_mock.complete_chat.reset_mock()
    listener_mock.on_chat_message.reset_mock()
    listener_mock.on_error.reset_mock()

    should_continue = session.process_input(":r")
    assert should_continue

    assistant_mock.complete_chat.assert_not_called()
    listener_mock.on_chat_message.assert_not_called()
    listener_mock.on_error.assert_not_called()
    listener_mock.on_chat_rerun.assert_called_once_with(False)


def test_openai_error():
    assistant_mock, listener_mock, session = setup_session()

    error = CompletionError("error message")
    assistant_mock.complete_chat.side_effect = error

    user_input = "user message"
    should_continue = session.process_input(user_input)
    assert should_continue

    user_message = {"role": "user", "content": user_input}
    listener_mock.on_chat_message.assert_has_calls([mock.call(user_message)])
    listener_mock.on_error.assert_called_once_with(error)

    # Re-run should work
    assistant_mock.complete_chat.reset_mock()
    listener_mock.on_chat_message.reset_mock()
    listener_mock.on_error.reset_mock()

    assistant_mock.complete_chat.side_effect = None
    assistant_mock.complete_chat.return_value = [MessageDeltaEvent("assistant message")]

    should_continue = session.process_input(":r")
    assert should_continue

    assistant_mock.complete_chat.assert_called_once_with(
        [system_message, user_message],
        stream=True,
    )
    listener_mock.on_chat_message.assert_has_calls(
        [
            mock.call({"role": "assistant", "content": "assistant message"}),
        ]
    )


def test_stream():
    assistant_mock, listener_mock, session = setup_session()
    assistant_message = "assistant message"
    assistant_mock.complete_chat.return_value = (
        MessageDeltaEvent(tok) for tok in list(assistant_message)
    )

    response_streamer_mock = listener_mock.response_streamer.return_value

    session.process_input("user message")

    response_streamer_mock.assert_has_calls(
        [mock.call.on_next_token(token) for token in assistant_message]
    )


def test_response_without_citations():
    assistant_mock, listener_mock, session = setup_session()
    response_streamer_mock = listener_mock.response_streamer.return_value

    assistant_mock.complete_chat.return_value = [MessageDeltaEvent("assistant message")]

    session.process_input("user message")

    response_streamer_mock.on_citations.assert_not_called()
    listener_mock.on_chat_response.assert_called_once_with(
        mock.ANY, mock.ANY, None, citations=[]
    )
    assert session.message_citations[-1] is None


def test_citations_follow_response_but_never_rewrite_body():
    assistant_mock, listener_mock, session = setup_session()
    response_streamer_mock = listener_mock.response_streamer.return_value

    citations = [
        Citation(
            url="https://example.com/source",
            title="Example Source",
            start_index=0,
            end_index=9,
        )
    ]
    assistant_mock.complete_chat.return_value = [
        MessageDeltaEvent("assistant message"),
        # Late-arriving citations, emitted after all text deltas
        CitationsEvent(citations),
    ]

    session.process_input("user message")

    # The streamer receives the citations after the full body
    response_streamer_mock.on_citations.assert_called_once_with(citations)

    # Listeners get the citations alongside the response
    listener_mock.on_chat_response.assert_called_once_with(
        mock.ANY,
        {"role": "assistant", "content": "assistant message"},
        None,
        citations=citations,
    )

    # The message persisted in the session stays pure body text
    saved_message = session.messages[-1]
    assert saved_message == {"role": "assistant", "content": "assistant message"}
    assert "citations" not in saved_message
    assert session.message_citations[-1] == citations
    assert len(session.message_citations) == len(session.messages)


def test_rerun_replaces_citations_instead_of_mixing_them():
    assistant_mock, listener_mock, session = setup_session()
    response_streamer_mock = listener_mock.response_streamer.return_value

    old_citations = [
        Citation(url="https://example.com/old", title="Old", start_index=0, end_index=1)
    ]
    new_citations = [
        Citation(url="https://example.com/new", title="New", start_index=2, end_index=3)
    ]

    assistant_mock.complete_chat.return_value = [
        MessageDeltaEvent("first response"),
        CitationsEvent(old_citations),
    ]
    session.process_input("user message")
    assert session.message_citations[-1] == old_citations

    assistant_mock.complete_chat.reset_mock()
    listener_mock.reset_mock()
    response_streamer_mock.reset_mock()

    assistant_mock.complete_chat.return_value = [
        MessageDeltaEvent("second response"),
        CitationsEvent(new_citations),
    ]
    should_continue = session.process_input(":r")
    assert should_continue

    # Only the freshly produced citations remain, aligned with the new message
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "second response",
    }
    assert session.message_citations[-1] == new_citations
    assert all(
        citation != old_citations[0] for citation in session.message_citations[-1]
    )
    assert len(session.message_citations) == len(session.messages)
    response_streamer_mock.on_citations.assert_called_once_with(new_citations)


def test_clear_resets_citations():
    assistant_mock, listener_mock, session = setup_session()

    assistant_mock.complete_chat.return_value = [
        MessageDeltaEvent("assistant message"),
        CitationsEvent(
            [Citation(url="https://example.com", title="T", start_index=0, end_index=1)]
        ),
    ]
    session.process_input("user message")
    assert session.message_citations[-1] is not None

    should_continue = session.process_input(":c")
    assert should_continue

    assert session.message_citations == [None] * len(session.messages)
    assert all(entry is None for entry in session.message_citations)
