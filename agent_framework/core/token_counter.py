"""
Token Counter - 粗略估算 token 数

不同模型 tokenizer 不同，但我们只需要"估算"来决定何时压缩。
用字符数 / 4 作为近似（英文），中文按 / 1.5 算。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


def _estimate_text_tokens(text: str) -> int:
    """估算一段文本的 token 数"""
    if not text:
        return 0

    # 统计中文字符数
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars

    # 中文 ~1.5 字符/token，英文 ~4 字符/token
    return int(chinese_chars / 1.5 + other_chars / 4) + 1


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    """估算单条 message 的 token 数"""
    tokens = 4  # role 等固定开销

    content = message.get("content")
    if isinstance(content, str):
        tokens += _estimate_text_tokens(content)
    elif isinstance(content, list):
        # 多模态 content
        for block in content:
            if isinstance(block, dict):
                tokens += _estimate_text_tokens(str(block.get("text", "")))

    # tool_calls
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        fn = tc.get("function", {})
        tokens += _estimate_text_tokens(fn.get("name", ""))
        args = fn.get("arguments", "")
        if isinstance(args, str):
            tokens += _estimate_text_tokens(args)
        elif isinstance(args, dict):
            tokens += _estimate_text_tokens(json.dumps(args))

    # tool 消息的 tool_call_id
    if message.get("tool_call_id"):
        tokens += 4

    return tokens


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """估算整个 messages 列表的 token 数"""
    return sum(estimate_message_tokens(m) for m in messages)