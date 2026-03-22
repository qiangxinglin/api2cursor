"""适配器公共辅助函数

收敛多个适配器都在重复实现的 CC 格式构建逻辑：
- CC 消息/Usage/Tool Call/Stream Chunk 的标准构造
- 内容扁平化、JSON 安全解析、工具输出序列化
"""

from __future__ import annotations

import json
from typing import Any

from utils.http import gen_id

JsonDict = dict[str, Any]


# ═══════════════════════════════════════════════════════════
#  CC 格式标准构造
# ═══════════════════════════════════════════════════════════


def build_cc_message(
    content_text: str,
    reasoning_text: str = '',
    tool_calls: list[JsonDict] | None = None,
) -> JsonDict:
    """构造标准的 CC assistant 消息。"""
    message: JsonDict = {
        'role': 'assistant',
        'content': content_text or None,
    }
    if reasoning_text:
        message['reasoning_content'] = reasoning_text
    if tool_calls:
        message['tool_calls'] = tool_calls
    return message


def build_cc_usage(input_tokens: int, output_tokens: int) -> JsonDict:
    """构造标准的 CC usage 字典。"""
    return {
        'prompt_tokens': input_tokens,
        'completion_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
    }


def build_cc_tool_call(
    call_id: str,
    name: str,
    arguments: str,
    *,
    index: int | None = None,
) -> JsonDict:
    """构造标准的 CC tool_call 结构。"""
    tc: JsonDict = {
        'id': call_id or gen_id('call_'),
        'type': 'function',
        'function': {
            'name': name,
            'arguments': arguments,
        },
    }
    if index is not None:
        tc['index'] = index
    return tc


def make_cc_chunk(
    chunk_id: str,
    delta: JsonDict,
    finish_reason: str | None = None,
    model: str = '',
) -> JsonDict:
    """构造标准的 CC 流式 chunk。"""
    choice: JsonDict = {'index': 0, 'delta': delta}
    if finish_reason:
        choice['finish_reason'] = finish_reason
    return {
        'id': chunk_id,
        'object': 'chat.completion.chunk',
        'model': model,
        'choices': [choice],
    }


def build_cc_response(
    response_id: str,
    message: JsonDict,
    finish_reason: str,
    usage: JsonDict,
    model: str = '',
) -> JsonDict:
    """构造标准的 CC 非流式响应。"""
    return {
        'id': response_id,
        'object': 'chat.completion',
        'model': model,
        'choices': [{
            'index': 0,
            'message': message,
            'finish_reason': finish_reason,
        }],
        'usage': usage,
    }


# ═══════════════════════════════════════════════════════════
#  通用文本/JSON 处理
# ═══════════════════════════════════════════════════════════


def extract_text(content: Any) -> str:
    """从多种内容格式中提取并拼接纯文本。

    支持字符串、内容块列表（OpenAI/Anthropic/Responses 风格）。
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ''

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            part_type = part.get('type', '')
            if part_type in ('text', 'output_text', 'input_text'):
                parts.append(part.get('text', ''))
            elif part_type == 'refusal':
                parts.append(part.get('refusal', ''))
            elif 'text' in part and not part_type:
                parts.append(part['text'])
    return '\n'.join(parts) if parts else ''


def parse_json_safe(text: Any, fallback: Any = None) -> Any:
    """安全解析 JSON，失败时返回 fallback。"""
    if not isinstance(text, str):
        return text if text is not None else (fallback if fallback is not None else {})
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return fallback if fallback is not None else {}


def stringify_content(content: Any) -> str:
    """将任意内容序列化为字符串。"""
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    return json.dumps(content, ensure_ascii=False)
