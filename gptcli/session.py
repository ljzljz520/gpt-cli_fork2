from abc import abstractmethod
from gptcli.assistant import Assistant
from gptcli.completion import (
    Citation,
    Message,
    CompletionError,
    BadRequestError,
    ToolCallEvent,
    UsageEvent,
)
from typing import List, Optional


class ResponseStreamer:
    def __enter__(self) -> "ResponseStreamer":
        return self

    def on_next_token(self, token: str):
        pass

    def on_thinking_token(self, token: str):
        pass

    def on_tool_call(self, tool_call: ToolCallEvent):
        pass

    def on_citations(self, citations: List[Citation]):
        pass

    def __exit__(self, *args):
        pass


class ChatListener:
    def on_chat_start(self):
        pass

    def on_chat_clear(self):
        pass

    def on_chat_rerun(self, success: bool):
        pass

    def on_error(self, error: Exception):
        pass

    def response_streamer(self) -> ResponseStreamer:
        return ResponseStreamer()

    def on_chat_message(self, message: Message):
        pass

    def on_chat_response(
        self,
        messages: List[Message],
        response: Message,
        usage: Optional[UsageEvent] = None,
        citations: Optional[List[Citation]] = None,
    ):
        pass


class UserInputProvider:
    @abstractmethod
    def get_user_input(self) -> str:
        pass


class InvalidArgumentError(Exception):
    def __init__(self, message: str):
        self.message = message


COMMAND_CLEAR = (":clear", ":c")
COMMAND_QUIT = (":quit", ":q")
COMMAND_RERUN = (":rerun", ":r")
COMMAND_HELP = (":help", ":h", ":?")
ALL_COMMANDS = [*COMMAND_CLEAR, *COMMAND_QUIT, *COMMAND_RERUN, *COMMAND_HELP]
COMMANDS_HELP = """
Commands:
- `:clear` / `:c` / Ctrl+C - Clear the conversation.
- `:quit` / `:q` / Ctrl+D - Quit the program.
- `:rerun` / `:r` / Ctrl+R - Re-run the last message.
- `:help` / `:h` / `:?` - Show this help message.
"""


class ChatSession:
    def __init__(
        self,
        assistant: Assistant,
        listener: ChatListener,
        stream: bool = True,
    ):
        self.assistant = assistant
        self.messages: List[Message] = assistant.init_messages()
        # Citations are kept outside of ``messages`` so the body sent back to
        # the model is never rewritten with footnote markup. Entries are aligned
        # one-to-one with ``messages`` (``None`` for messages without citations).
        self.message_citations: List[Optional[List[Citation]]] = [None] * len(
            self.messages
        )
        self.user_prompts: List[Message] = []
        self.listener = listener
        self.stream = stream

    def _clear(self):
        self.messages = self.assistant.init_messages()
        self.message_citations = [None] * len(self.messages)
        self.user_prompts = []
        self.listener.on_chat_clear()

    def _rerun(self):
        if len(self.user_prompts) == 0:
            self.listener.on_chat_rerun(False)
            return

        if self.messages[-1]["role"] == "assistant":
            self.messages = self.messages[:-1]
            self.message_citations = self.message_citations[:-1]

        self.listener.on_chat_rerun(True)
        self._respond()

    def _respond(self) -> bool:
        """
        Respond to the user's input and return whether the assistant's response was saved.
        """
        next_response: str = ""
        citations: List[Citation] = []
        usage: Optional[UsageEvent] = None
        try:
            completion_iter = self.assistant.complete_chat(
                self.messages, stream=self.stream
            )

            with self.listener.response_streamer() as stream:
                for event in completion_iter:
                    if event.type == "message_delta":
                        next_response += event.text
                        stream.on_next_token(event.text)
                    elif event.type == "thinking_delta":
                        stream.on_thinking_token(event.text)
                    elif event.type == "tool_call":
                        stream.on_tool_call(event)
                    elif event.type == "usage":
                        usage = event
                    elif event.type == "citations":
                        # A completion emits at most one citations event, and it
                        # is rebuilt from scratch on every run, so a rerun can
                        # never mix in citations from the previous response.
                        citations = event.citations

                # Citations may arrive after the last text delta; render them
                # only once the full response body has been streamed.
                if citations:
                    stream.on_citations(citations)

        except KeyboardInterrupt:
            # If the user interrupts the chat completion, we'll just return what we have so far
            pass
        except BadRequestError as e:
            self.listener.on_error(e)
            return False
        except CompletionError as e:
            self.listener.on_error(e)
            return True

        next_message: Message = {"role": "assistant", "content": next_response}
        self.listener.on_chat_message(next_message)
        self.listener.on_chat_response(
            self.messages, next_message, usage, citations=citations
        )

        self.messages = self.messages + [next_message]
        self.message_citations = self.message_citations + [
            citations if citations else None
        ]
        return True

    def _add_user_message(self, user_input: str):
        user_message: Message = {"role": "user", "content": user_input}
        self.messages = self.messages + [user_message]
        self.message_citations = self.message_citations + [None]
        self.listener.on_chat_message(user_message)
        self.user_prompts.append(user_message)

    def _rollback_user_message(self):
        self.messages = self.messages[:-1]
        self.message_citations = self.message_citations[:-1]
        self.user_prompts = self.user_prompts[:-1]

    def _print_help(self):
        with self.listener.response_streamer() as stream:
            stream.on_next_token(COMMANDS_HELP)

    def process_input(self, user_input: str):
        """
        Process the user's input and return whether the session should continue.
        """
        if user_input in COMMAND_QUIT:
            return False
        elif user_input in COMMAND_CLEAR:
            self._clear()
            return True
        elif user_input in COMMAND_RERUN:
            self._rerun()
            return True
        elif user_input in COMMAND_HELP:
            self._print_help()
            return True

        self._add_user_message(user_input)
        response_saved = self._respond()
        if not response_saved:
            self._rollback_user_message()

        return True

    def loop(self, input_provider: UserInputProvider):
        self.listener.on_chat_start()
        while self.process_input(input_provider.get_user_input()):
            pass
