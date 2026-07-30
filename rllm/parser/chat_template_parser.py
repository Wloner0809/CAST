import json
import logging
import re
from copy import deepcopy

import torch

from rllm.tools.tool_base import Tool, ToolCall, ToolOutput

from .utils import PARSER_TEST_MESSAGES

logger = logging.getLogger(__name__)


class ChatTemplateParser:
    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.generation_prompt = self._get_generation_prompt(tokenizer)

    def _get_generation_prompt(self, tokenizer):
        messages = [{"role": "assistant", "content": ""}]

        with_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        without_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        generation_prompt = with_prompt[len(without_prompt) :]

        return generation_prompt

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        if self.processor is not None:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        else:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def parse_completion(self, completion_ids: list[int]):
        raise NotImplementedError("ChatTemplateParser does not support parse_completion")

    def verify_equivalence(self, messages, verbose=True):
        """Verify that parsing messages together is equivalent to parsing them individually.

        Args:
            messages (list): List of message dictionaries to test
            verbose (bool): Whether to print detailed information about the test

        Returns:
            bool: True if the equivalence check passes, False otherwise

        Raises:
            AssertionError: If the equivalence check fails and verbose is True
        """
        # Parse all messages together
        batch_result = self.parse(messages)

        # Parse each message individually and concatenate
        individual_results = []
        for message in messages:
            individual_results.append(self.parse([message]))

        concatenated_result = "".join(individual_results)

        # Check if results are equivalent
        is_equivalent = batch_result == concatenated_result

        if verbose and not is_equivalent:
            print("Equivalence check failed!")
            print("Batch parsing result:")
            print(batch_result)
            print("\nConcatenated individual parsing result:")
            print(concatenated_result)
            raise AssertionError("Parser failed equivalence check. See above for details.")

        return is_equivalent

    @classmethod
    def get_parser(cls, tokenizer, processor=None, disable_thinking=False) -> "ChatTemplateParser":
        """Factory method to get the appropriate parser based on a string identifier.

        Args:
            parser_type (str): String identifier for the parser type
            tokenizer: The tokenizer to use with the parser
            disable_thinking: Whether generation prompt will disable thinking.

        Returns:
            ChatTemplateParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Determine parser type based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            logger.info(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                if "deepseek-math-v2" in model_name or "deepseek-v3.2-exp" in model_name:
                    logger.info(f"Using DeepSeekV32ExpChatTemplateParser for {tokenizer.name_or_path}")
                    return DeepSeekV32ExpChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
                else:
                    logger.info(f"Using DeepseekQwenChatTemplateParser for {tokenizer.name_or_path}")
                    return DeepseekQwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                logger.info(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                return QwenChatTemplateParser(tokenizer, processor=processor, disable_thinking=disable_thinking)
            elif "llama" in model_name:
                logger.info(f"Using LlamaChatTemplateParser for {tokenizer.name_or_path}")
                return LlamaChatTemplateParser(tokenizer)
            elif "gpt-oss" in model_name or "imo" in model_name:
                logger.info(f"Using HarmonyChatTemplateParser for {tokenizer.name_or_path}")
                return HarmonyChatTemplateParser()

        # Default to the standard parser if no specific match
        parser = ChatTemplateParser(tokenizer, processor=processor)
        logger.info(f"No custom parser found. Using default ChatTemplateParser for {tokenizer.name_or_path}")
        assert parser.verify_equivalence(PARSER_TEST_MESSAGES), "Parser failed equivalence check"
        return parser

    def tokenize_and_mask(self, messages):
        try:
            last_assistant_idx = max(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except ValueError:
            raise ValueError("No assistant message found in chat_completions") from None

        prompt = self.parse(messages[:last_assistant_idx], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        response = self.parse([messages[last_assistant_idx]], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
        response = response[len(self.generation_prompt) :].rstrip("\n")  # handle qwen trailing newline from eot token
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        response_mask = [1] * len(response_ids)

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        return prompt_ids, response_ids, response_mask

    def tokenize_and_mask_cumulative(self, messages, skip_trailing_generation_prompt=False):
        """Tokenize a conversation into prompt + response tokens with masks.

        Converts a multi-turn conversation into training format by tokenizing
        each message independently (avoiding cross-message retokenization issues).
        Assistant messages get mask=1 (trainable), other messages get mask=0.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            skip_trailing_generation_prompt: If True, don't add generation prompt
                after the very last message when it's a non-assistant message.
                Useful when the trajectory ended (done=True) and there's no next turn.

        Returns:
            Tuple of (prompt_ids, response_ids, response_mask) as torch tensors.
        """
        response_ids = []
        response_mask = []

        try:
            first_assistant_idx = next(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except StopIteration:
            raise ValueError("No assistant message found in chat_completions") from None

        prompt = self.parse(messages[:first_assistant_idx], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        last_idx = len(messages) - 1
        for i in range(first_assistant_idx, len(messages)):
            is_asst = messages[i]["role"] == "assistant"
            if is_asst:
                response = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
                response = response[len(self.generation_prompt) :]
                ids = self.tokenizer.encode(response, add_special_tokens=False)
                response_ids.extend(ids)
                response_mask.extend([1] * len(ids))
            else:
                # For non-assistant messages, add generation_prompt to prepare for
                # the next assistant turn — unless this is the last message and
                # skip_trailing_generation_prompt is set (trajectory already ended).
                add_gen = not (skip_trailing_generation_prompt and i == last_idx)
                response = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=add_gen, accumulate_reasoning=False)
                ids = self.tokenizer.encode(response, add_special_tokens=False)
                response_ids.extend(ids)
                response_mask.extend([0] * len(ids))

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        return prompt_ids, response_ids, response_mask


class DeepseekQwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        self.generation_prompt = self.assistant_token + "<think>\n"

        from rllm.parser.tool_parser import R1ToolParser

        self.tool_parser = R1ToolParser()

    def parse(self, messages: list[dict], add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list[Tool | dict] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        tools = tools or []
        tools_prompt_str = ""
        if tools:
            try:
                tool_schema_strs = []
                for tool in tools:
                    if isinstance(tool, Tool):
                        tool_schema_str = json.dumps(tool.json)
                    elif isinstance(tool, dict):
                        tool_schema_str = json.dumps(tool)
                    else:
                        tool_schema_str = tool
                    tool_schema_strs.append(tool_schema_str)
                tools_schema_str = "\n".join(tool_schema_strs)
                tools_prompt_str = self.tool_parser.get_tool_prompt(tools_schema_str)
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Failed to format tools: {e}")

        result = ""

        if is_first_msg:
            result += self.bos_token

        if is_first_msg and messages[0]["role"] != "system" and tools_prompt_str:
            result += self.system_token + tools_prompt_str

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message, tools_prompt_str)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message, accumulate_reasoning=accumulate_reasoning)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message, tools_prompt_str=""):
        content = message["content"]

        if "# Tools" not in content and tools_prompt_str:
            content += tools_prompt_str

        return self.system_token + content

    def parse_user(self, message):
        return self.user_token + message["content"]

    def parse_assistant(self, message, accumulate_reasoning=False):
        # IMPORTANT: Do NOT .strip() content here. Stripping removes trailing whitespace/newlines
        # that exist in the original completion_ids. When assemble_steps compares
        # accumulated_sequence (built from original completion_ids) against re-tokenized
        # prompt_ids, any stripped characters cause a token mismatch at the step boundary.
        # Example: model generates "action\n<eos>", content stored as "action\n",
        # .strip() would produce "action", losing the \n token → mismatch.
        content = message.get("content", None) or ""
        reasoning = (message.get("reasoning", None) or "").strip()
        tool_calls = message.get("tool_calls", None) or []

        if not accumulate_reasoning:
            return self.assistant_token + content + self.eos_token
        elif not reasoning:
            # When accumulate_reasoning=True but no separate reasoning field:
            # The content was decoded from completion_ids (skip_special_tokens=True)
            # and does NOT contain the <think>\n prefix (that was part of generation_prompt).
            # We must re-add <think>\n here so that:
            #   tokenize(assistant_token + <think>\n + content + eos)
            # matches the original:
            #   tokenize(generation_prompt) + completion_ids
            # Without this, assemble_steps sees a mismatch at every step boundary.
            return self.assistant_token + "<think>\n" + content + self.eos_token
        else:
            result = self.assistant_token

            if reasoning and accumulate_reasoning:
                result += "<think>\n" + reasoning
                if content:
                    result += "\n</think>\n\n"

            if content:
                result += content
                if tool_calls:
                    result += "\n"

            if tool_calls:
                try:
                    tool_calls_strs = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, ToolCall):
                            tool_call_dict = tool_call.to_dict()
                        elif isinstance(tool_call, dict) and "function" in tool_call:
                            tool_call_dict = tool_call["function"]
                        else:
                            tool_call_dict = tool_call
                        # Avoid mutating original message structures by parsing into a local variable
                        arguments_obj = tool_call_dict.get("arguments")
                        if isinstance(arguments_obj, str):
                            try:
                                arguments_obj = json.loads(arguments_obj)
                            except json.JSONDecodeError:
                                pass
                        tool_call_json = f"```json\n{json.dumps(arguments_obj)}\n```"
                        tool_call_str = f"{self.tool_parser.tool_call_begin}function{self.tool_parser.tool_sep}{tool_call_dict['name']}\n{tool_call_json}\n{self.tool_parser.tool_call_end}"
                        tool_calls_strs.append(tool_call_str)
                    joined_calls_str = "\n".join(tool_calls_strs)
                    tool_calls_str = f"{self.tool_parser.tool_calls_begin}\n{joined_calls_str}\n{self.tool_parser.tool_calls_end}"
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logger.error(f"Failed to format tool calls: {e}")
                    tool_calls_str = ""

                result += tool_calls_str

            result += self.eos_token
            return result

    def parse_tool(self, message):
        tool_outputs: list[ToolOutput | dict] = message.get("tool_outputs", [])

        if not tool_outputs:
            return self.user_token + self.tool_parser.tool_output_begin + "\n" + message["content"] + "\n" + self.tool_parser.tool_output_end

        else:
            try:
                tool_outputs_strs = []
                for tool_output in tool_outputs:
                    if not isinstance(tool_output, ToolOutput):
                        tool_output = ToolOutput(**tool_output)
                    tool_output_str = f"{self.tool_parser.tool_output_begin}\n{str(tool_output)}\n{self.tool_parser.tool_output_end}"
                    tool_outputs_strs.append(tool_output_str)
                tool_outputs_str = "\n".join(tool_outputs_strs)
            except Exception as e:
                logger.error(f"Failed to format tool outputs: {e}")
                tool_outputs_str = ""

            return self.user_token + tool_outputs_str

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            reasoning = reasoning.strip()
            content = content.strip()
        else:
            # DeepSeekQwen should always have reasoning
            reasoning = completion_text.strip()
            content = ""

        if content:
            # parse tool calls from content
            tool_calls = self.tool_parser.parse(content)
            begin_pattern = re.escape(self.tool_parser.tool_call_begin)
            end_pattern = re.escape(self.tool_parser.tool_call_end)
            wrapper_begin_pattern = re.escape(self.tool_parser.tool_calls_begin)
            wrapper_end_pattern = re.escape(self.tool_parser.tool_calls_end)
            content = re.sub(f"{begin_pattern}.*?{end_pattern}", "", content, flags=re.DOTALL)
            content = re.sub(f"{wrapper_begin_pattern}.*?{wrapper_end_pattern}", "", content, flags=re.DOTALL)
            content = content.strip()
        else:
            tool_calls = []

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        }


class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, processor=None, disable_thinking=False):
        super().__init__(tokenizer, processor=processor)
        self.disable_thinking = disable_thinking
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        if disable_thinking:
            self.assistant_token += "<think>\n\n</think>\n\n"
        self.generation_prompt = self.assistant_token
        self.image_token = "<|image_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"

        from rllm.parser.tool_parser import QwenToolParser

        self.tool_parser = QwenToolParser()
        self.im_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.endoftext_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.stop_token_ids = [
            token_id
            for token_id in (
                tokenizer.eos_token_id,
                self.im_end_token_id,
                self.endoftext_token_id,
            )
            if token_id is not None and token_id != tokenizer.unk_token_id
        ]

        # Pre-compute special token strings for stripping in parse_completion
        self.endoftext_str = self.tokenizer.decode([self.endoftext_token_id]) if self.endoftext_token_id is not None else None
        self.im_start_str = self.tokenizer.decode([self.im_start_token_id]) if self.im_start_token_id is not None else None

        # Build mapping from stop_token_id → eot string (token + \n) for parse_assistant.
        # This allows parse_assistant to use the correct eot based on which stop token
        # the model actually generated (e.g., Qwen2.5 may use <|endoftext|> instead of <|im_end|>).
        # When <|im_start|> triggers a stop, the generation was degenerate (model tried to
        # start a new role turn mid-completion). We use <|im_end|>\n as the eot since the
        # completion is effectively terminated at that point and the next turn's prompt will
        # add the proper <|im_start|> structure.
        self._stop_token_eot_map = {}
        if self.im_end_token_id is not None:
            self._stop_token_eot_map[self.im_end_token_id] = "<|im_end|>\n"
        if self.endoftext_token_id is not None:
            self._stop_token_eot_map[self.endoftext_token_id] = "<|endoftext|>\n"

    def parse(self, messages: list[dict], add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list[Tool] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        tools = tools or []
        tools_prompt_str = ""
        if tools:
            try:
                tool_schema_strs = []
                for tool in tools:
                    if isinstance(tool, Tool):
                        tool_schema_str = json.dumps(tool.json)
                    elif isinstance(tool, dict):
                        tool_schema_str = json.dumps(tool)
                    else:
                        tool_schema_str = tool
                    tool_schema_strs.append(tool_schema_str)
                tools_schema_str = "\n".join(tool_schema_strs)
                tools_prompt_str = self.tool_parser.get_tool_prompt(tools_schema_str)
            except Exception as e:
                logger.error(f"Failed to format tools: {e}")

        result = ""

        # if the first message is not a system message, add the system message
        if is_first_msg and messages[0]["role"] != "system":
            result += self.system_token + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + tools_prompt_str + self.eot_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message, tools_prompt_str)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message, accumulate_reasoning=accumulate_reasoning)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message, tools_prompt_str=""):
        content = message["content"]

        if "# Tools" not in content and tools_prompt_str:
            content += tools_prompt_str

        return self.system_token + content + self.eot_token

    def parse_user(self, message):
        if "images" in message and message["images"] is not None:
            assert isinstance(message["images"], list), "images must be a list"
            n_imgs = len(message["images"])
            content = message["content"]
            if message["content"].startswith("<image>"):
                content = content[len("<image>") :]
            vision_tokens = (self.vision_start_token + self.image_token + self.vision_end_token) * n_imgs
            return self.user_token + vision_tokens + content + self.eot_token

        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message, accumulate_reasoning=False):
        # IMPORTANT: Do NOT .strip() content here. Stripping removes trailing whitespace/newlines
        # that exist in the original completion_ids. When assemble_steps compares
        # accumulated_sequence (built from original completion_ids) against re-tokenized
        # prompt_ids, any stripped characters cause a token mismatch at the step boundary.
        # Example: model generates "action\n<|im_end|>", content stored as "action\n",
        # .strip() would produce "action", losing the \n token → mismatch.
        content = message.get("content", None) or ""
        reasoning = (message.get("reasoning", None) or "").strip()
        tool_calls = message.get("tool_calls", None) or []

        # Use actual stop token if available (propagated from completion_ids via
        # _stop_token_id in the message dict), otherwise fall back to the default
        # eot_token (<|im_end|>\n). This is needed because Qwen2.5 may generate
        # <|endoftext|> instead of <|im_end|>, and parse_assistant must use the
        # matching eot string so assemble_steps prefix comparison succeeds.
        stop_token_id = message.get("_stop_token_id")
        eot = self._stop_token_eot_map.get(stop_token_id, self.eot_token) if stop_token_id is not None else self.eot_token

        if not reasoning and not tool_calls:
            return self.assistant_token + content + eot

        else:
            result = self.assistant_token
            if reasoning and accumulate_reasoning:
                result += "<think>\n" + reasoning
                if content:
                    result += "\n</think>\n\n"

            if content:
                result += content
                if tool_calls:
                    result += "\n"

            if tool_calls:
                try:
                    tool_calls_strs = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, ToolCall):
                            tool_call_dict = tool_call.to_dict()
                        elif isinstance(tool_call, dict) and "function" in tool_call:
                            tool_call_dict = tool_call["function"]
                        else:
                            tool_call_dict = tool_call
                        arguments_obj = tool_call_dict.get("arguments")
                        if isinstance(arguments_obj, str):
                            try:
                                arguments_obj = json.loads(arguments_obj)
                            except json.JSONDecodeError:
                                pass
                        tool_call_for_dump = dict(tool_call_dict)
                        if arguments_obj is not None:
                            tool_call_for_dump["arguments"] = arguments_obj
                        tool_call_str = f"{self.tool_parser.tool_call_begin}\n{json.dumps(tool_call_for_dump)}\n{self.tool_parser.tool_call_end}"
                        tool_calls_strs.append(tool_call_str)
                    tool_calls_str = "\n".join(tool_calls_strs)
                except Exception as e:
                    logger.error(f"Failed to format tool calls: {e}")
                    tool_calls_str = ""

                result += tool_calls_str

            result += eot
            return result

    def parse_tool(self, message):
        tool_outputs: list[ToolOutput | dict] = message.get("tool_outputs", [])

        if not tool_outputs:
            return self.user_token + self.tool_parser.tool_output_begin + "\n" + message["content"] + "\n" + self.tool_parser.tool_output_end + self.eot_token

        else:
            try:
                tool_outputs_strs = []
                for tool_output in tool_outputs:
                    if not isinstance(tool_output, ToolOutput):
                        tool_output = ToolOutput(**tool_output)
                    tool_output_str = f"{self.tool_parser.tool_output_begin}\n{str(tool_output)}\n{self.tool_parser.tool_output_end}"
                    tool_outputs_strs.append(tool_output_str)
                tool_outputs_str = "\n".join(tool_outputs_strs)
            except Exception as e:
                logger.error(f"Failed to format tool outputs: {e}")
                tool_outputs_str = ""

            return self.user_token + tool_outputs_str + self.eot_token

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            if content.endswith(self.eot_token):
                content = content[: -len(self.eot_token)]
            # Also strip <|endoftext|> which Qwen2.5 may generate instead of <|im_end|>
            if self.endoftext_str and content.endswith(self.endoftext_str):
                content = content[: -len(self.endoftext_str)]
            reasoning = reasoning.strip()
            content = content.strip()
        elif not self.disable_thinking:
            # generation was cut short during reasoning
            reasoning = completion_text
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            if reasoning.endswith(self.eos_token):
                reasoning = reasoning[: -len(self.eos_token)]
            if reasoning.endswith(self.eot_token):
                reasoning = reasoning[: -len(self.eot_token)]
            # Strip <|endoftext|> from cut-short reasoning (Qwen2.5 may generate it)
            if self.endoftext_str and reasoning.endswith(self.endoftext_str):
                reasoning = reasoning[: -len(self.endoftext_str)]
            reasoning = reasoning.strip()
            content = ""
        else:
            # thinking is disabled, so everything is content
            reasoning = ""
            content = completion_text
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            if content.endswith(self.eot_token):
                content = content[: -len(self.eot_token)]
            # Also strip <|endoftext|> which Qwen2.5 may generate instead of <|im_end|>
            if self.endoftext_str and content.endswith(self.endoftext_str):
                content = content[: -len(self.endoftext_str)]
            content = content.strip()

        if content:
            # parse tool calls from content
            tool_calls = self.tool_parser.parse(content)
            begin_pattern = re.escape(self.tool_parser.tool_call_begin)
            end_pattern = re.escape(self.tool_parser.tool_call_end)
            content = re.sub(f"{begin_pattern}.*?{end_pattern}", "", content, flags=re.DOTALL)
            content = content.strip()
        else:
            tool_calls = []

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        }

    def process_image_data(self, messages):
        from qwen_vl_utils import fetch_image

        messages = deepcopy(messages)
        image_data = []
        for message in messages:
            if "images" in message and message["images"] is not None:
                assert isinstance(message["images"], list), "images must be a list"
                images = message["images"]
                if not images or images[0] is None:
                    continue
                for image in images:
                    image_dict = image if isinstance(image, dict) else {"image": image}
                    processed_image = fetch_image(image_dict, image_patch_size=self.processor.image_processor.patch_size)  # PIL.Image.Image
                    image_data.append(processed_image)
        return image_data

class LlamaChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = "<|begin_of_text|>"
        self.system_token = "<|start_header_id|>system<|end_header_id|>\n\n"
        self.user_token = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.eot_token = "<|eot_id|>"
        self.generation_prompt = self.assistant_token

        # tool tokens
        self.tool_start_token = "<|start_header_id|>tool<|end_header_id|>\n\n"
        self.tool_end_token = "<|eot_id|>"
        self.tool_response_start_token = "<|start_header_id|>tool_response<|end_header_id|>\n\n"
        self.tool_response_end_token = "<|eot_id|>"

        # TODO: add tool parser for llama

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eot_token

    def parse_tool(self, message):
        return self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token

    def parse_completion(self, completion_ids):
        # TODO: add parse_completion for llama
        raise NotImplementedError("LLamaChatTemplateParser does not support parse_completion")


class HarmonyChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer=None):
        from openai_harmony import (
            HarmonyEncodingName,
            load_harmony_encoding,
        )

        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.generation_prompt = "<|start|>assistant"
        self.stop_sequences = [200002, 199999, 200012]  # <|endoftext|>, <|return|>, <|call|>

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        return self.parse_prompt_from_messages(messages, add_generation_prompt=add_generation_prompt, is_first_msg=is_first_msg, **kwargs)

    def parse_prompt_from_messages(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs):
        from openai_harmony import Conversation, DeveloperContent, Message, ReasoningEffort, RenderConversationConfig, Role, SystemContent

        # messages is a list[dict], where each dict is of the following structure:
        # {
        #     "role": str,
        #     "content": str,
        #     "reasoning": str, # optional
        # }

        messages = deepcopy(messages)
        harmony_messages: list[Message] = []

        if is_first_msg:
            # 1. system prompt
            reasoning_effort = ReasoningEffort(kwargs.get("reasoning_effort", "medium").capitalize())
            system_message = SystemContent.new().with_reasoning_effort(reasoning_effort)
            harmony_messages.append(Message.from_role_and_content(Role.SYSTEM, system_message))

            # 2. developer prompt
            if messages[0]["role"] == "system":
                instructions = messages.pop(0).get("content")
                developer_message = DeveloperContent.new().with_instructions(instructions)
                harmony_messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message))

        # 3. the rest of the messages
        for message in messages:
            if message["role"] == "user":
                harmony_messages.append(Message.from_role_and_content(Role.USER, message["content"]))
            elif message["role"] == "assistant":
                reasoning = message.get("reasoning", None)
                content = message.get("content", None)
                if reasoning:
                    harmony_messages.append(Message.from_role_and_content(Role.ASSISTANT, reasoning).with_channel("analysis"))
                if content:
                    harmony_messages.append(Message.from_role_and_content(Role.ASSISTANT, content).with_channel("final"))
            elif message["role"] == "tool":
                raise NotImplementedError("Tool messages are not supported yet")
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        conv = Conversation.from_messages(harmony_messages)
        accumulate_thinking = kwargs.get("accumulate_thinking", False)
        config = RenderConversationConfig(auto_drop_analysis=not accumulate_thinking)
        prompt_ids: list[int] = self.enc.render_conversation(conv, config)

        try:
            prompt: str = self.enc.decode_utf8(prompt_ids)
        except UnicodeDecodeError:
            prompt: str = self.enc.decode(prompt_ids)
            print(f"Warning: UnicodeDecodeError when decoding prompt: {prompt[:1000]}...")

        if add_generation_prompt:
            prompt += self.generation_prompt

        return prompt

    def parse_completion(self, completion_ids: list[int], **kwargs) -> dict[str, str | list]:
        from openai_harmony import Role

        # NOTE: harmony will throw an error if the sequence ends during the header (e.g., due to length)
        harmony_messages = self.enc.parse_messages_from_completion_tokens(completion_ids, role=Role.ASSISTANT)

        analysis = ""
        final = ""
        for message in harmony_messages:
            content = message.content[0].text
            channel = message.channel

            if channel == "analysis":
                analysis += content
            elif channel == "final":
                final += content

        # TODO: handle tool calls

        return {
            "content": final,
            "reasoning": analysis,
            "tool_calls": [],
        }


class DeepSeekV32ExpChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, disable_thinking=False):
        self.tokenizer = tokenizer
        self.disable_thinking = disable_thinking
        self.bos_token = "<｜begin▁of▁sentence｜>"
        self.eos_token = "<｜end▁of▁sentence｜>"
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        if disable_thinking:
            self.generation_prompt = self.assistant_token + "</think>"
        else:
            self.generation_prompt = self.assistant_token + "<think>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, tools: list[Tool | dict] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        if tools:
            raise NotImplementedError("Tools are not supported yet")

        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message, accumulate_reasoning=accumulate_reasoning)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt

        return result

    def parse_system(self, message):
        return self.system_token + message["content"]

    def parse_user(self, message):
        return self.user_token + message["content"]

    def parse_assistant(self, message, accumulate_reasoning=False):
        reasoning = message.get("reasoning", None)
        # IMPORTANT: Do NOT .strip() content — see DeepseekQwenChatTemplateParser
        # for detailed explanation of why stripping causes token mismatches.
        content = message.get("content", None)

        result = self.assistant_token
        if reasoning and accumulate_reasoning:
            result += "<think>" + reasoning
        if content:
            result += "</think>" + content
        result += self.eos_token

        return result

    def parse_tool(self, message):
        raise NotImplementedError("Tools are not supported yet")

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            reasoning = reasoning.strip()
            content = content.strip()
        elif not self.disable_thinking:
            # generation was cut short during reasoning
            reasoning = completion_text
            reasoning = reasoning.strip()
            content = ""
        else:
            # thinking is disabled, so everything is content
            reasoning = ""
            content = completion_text
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            content = content.strip()

        # TODO: handle tool calls

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": [],
        }
