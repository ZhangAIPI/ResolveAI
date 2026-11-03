"""Small checkpoint-specific processor settings; image evidence stays unchanged."""
from copy import deepcopy
import json
from pathlib import Path


def processor_options(model):
    path = Path(model) / "resolveai_processor.json"
    options = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(options, dict) or set(options) - {"size", "fix_mistral_regex", "use_fast"}:
        raise ValueError("Unsupported processor configuration")
    for key in ("fix_mistral_regex", "use_fast"):
        if key in options and type(options[key]) is not bool:
            raise ValueError("Processor flags must be booleans")
    if "size" in options:
        size = options["size"]
        if (not isinstance(size, dict) or set(size) != {"longest_edge"}
                or type(size["longest_edge"]) is not int or not 48 <= size["longest_edge"] <= 512):
            raise ValueError("Native longest edge must respect the declared image budget")
    return options


def typed_messages(messages, processor_name):
    messages = deepcopy(messages)
    for message in messages:
        content = message.get("content")
        # Pixtral's saved template concatenates its optional system message
        # with a string. Its user/assistant content supports typed media blocks.
        if processor_name == "PixtralProcessor" and message["role"] == "system":
            if isinstance(content, list):
                if any(block.get("type") != "text" for block in content):
                    raise ValueError("Images are not allowed in a Pixtral system message")
                message["content"] = "".join(block["text"] for block in content)
        elif isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
    if processor_name == "PixtralProcessor":
        # Its native template requires alternating speakers. Consecutive
        # messages from one speaker are one turn, in their original order.
        merged = []
        for message in messages:
            if merged and merged[-1]["role"] == message["role"]:
                if message["role"] == "system":
                    merged[-1]["content"] += "\n" + message["content"]
                else:
                    merged[-1]["content"].extend(
                        [{"type": "text", "text": "\n"}] + message["content"])
            else:
                merged.append(message)
        return merged
    return messages


def image_settings(processor):
    config = processor.image_processor.to_dict()
    return {key: config[key] for key in ("size", "max_image_size", "do_image_splitting", "crop_to_patches")
            if key in config}


def chat_inputs(processor, messages, *, tools=None, **template_options):
    messages = typed_messages(messages, type(processor).__name__)
    if type(processor).__name__ == "PixtralProcessor":
        # The checkpoint template expects a string system prompt, while
        # ProcessorMixin's tokenize=True path assumes typed blocks in all
        # roles. Use the checkpoint's documented text+ordered-images path.
        text = processor.apply_chat_template(messages, tools=tools,
            tokenize=False, add_generation_prompt=True, **template_options)
        images = [block["image"] for message in messages
                  if isinstance(message.get("content"), list)
                  for block in message["content"] if block.get("type") == "image"]
        return processor(text=text, images=images or None, return_tensors="pt")
    return processor.apply_chat_template(messages, tools=tools,
        tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", **template_options)
