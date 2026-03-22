"""统一中间格式与转换器接口

定义项目中所有 API 格式共用的中间表示和转换器协议：
- UnifiedRequest / UnifiedResponse: 统一的请求/响应数据结构
- InboundTransformer / OutboundTransformer: 入站/出站转换器接口
- StreamProcessor: 流式事件处理器接口
- ClientFormatter: 客户端响应格式化接口
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from flask import Response, jsonify

import settings
from utils.http import forward_request, gen_id, sse_response
from utils.request_logger import (
    append_client_event,
    append_upstream_event,
    attach_client_response,
    attach_error,
    attach_upstream_request,
    attach_upstream_response,
    finalize_turn,
    set_stream_summary,
)
from utils.usage_tracker import usage_tracker

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]


# ═══════════════════════════════════════════════════════════
#  统一数据模型
# ═══════════════════════════════════════════════════════════


@dataclass
class UnifiedUsage:
    """标准化的令牌用量统计。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def to_cc_dict(self) -> JsonDict:
        return {
            'prompt_tokens': self.input_tokens,
            'completion_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
        }

    def to_responses_dict(self) -> JsonDict:
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
        }

    @classmethod
    def from_cc_dict(cls, d: JsonDict) -> UnifiedUsage:
        return cls(
            input_tokens=d.get('prompt_tokens', 0),
            output_tokens=d.get('completion_tokens', 0),
            total_tokens=d.get('total_tokens', 0),
        )

    @classmethod
    def from_responses_dict(cls, d: JsonDict) -> UnifiedUsage:
        return cls(
            input_tokens=d.get('input_tokens', 0),
            output_tokens=d.get('output_tokens', 0),
            total_tokens=d.get('total_tokens', 0),
        )


# ═══════════════════════════════════════════════════════════
#  转换器接口
# ═══════════════════════════════════════════════════════════


class OutboundTransformer(Protocol):
    """出站转换器：将 CC 中间格式转换为上游后端格式。

    所有后端（OpenAI Chat / Responses / Anthropic / Gemini）各实现一套，
    内部复用各自现有的适配器函数。
    """

    def build_request(self, payload: JsonDict) -> JsonDict:
        """将 CC 格式请求体转换为上游格式请求体。"""
        ...

    def build_url(self, ctx: Any) -> str:
        """根据路由上下文构建上游请求 URL。"""
        ...

    def build_headers(self, ctx: Any) -> JsonDict:
        """根据路由上下文构建上游请求头。"""
        ...

    def parse_response(self, raw: JsonDict) -> JsonDict:
        """将上游非流式响应转换回 CC 格式。"""
        ...

    def create_stream_processor(self) -> StreamProcessor:
        """创建该后端对应的流式事件处理器。"""
        ...


class StreamProcessor(Protocol):
    """流式事件处理器接口。

    每个后端的 SSE 格式不同，StreamProcessor 封装了具体的迭代与转换逻辑，
    让通用流式处理器不必关心后端差异。
    """

    def iter_events(self, response: Any) -> Iterator:
        """从上游 HTTP 响应中迭代原始事件。"""
        ...

    def process_event(self, event: Any) -> list:
        """将单个上游事件转换为输出项列表。

        返回值通常是 list[JsonDict]（CC chunk），
        但 Anthropic→Responses 路径返回 list[str]（SSE 字符串）。
        """
        ...

    def extract_usage(self, event: Any) -> JsonDict | None:
        """从上游事件中提取用量信息（如果有的话）。"""
        ...

    def finalize(self) -> list:
        """流结束时产出的收尾项。"""
        ...


class ClientFormatter(Protocol):
    """客户端响应格式化器。

    根据客户端期望的 API 格式（CC 或 Responses），将通用的处理结果
    格式化为最终返回给客户端的形态。
    """

    def format_response(self, cc_response: JsonDict, model: str) -> JsonDict:
        """格式化非流式响应。"""
        ...

    def wrap_stream_item(self, item: Any) -> str:
        """将单个流式输出项包装为 SSE 字符串。"""
        ...

    def format_error(self, message: str) -> str:
        """构造流式错误消息。"""
        ...

    def format_done(self) -> str | None:
        """构造流结束标记（CC 返回 [DONE]，Responses 返回 None）。"""
        ...

    def start_events(self) -> list[str]:
        """流开始前的初始事件（Responses 返回 response.created）。"""
        ...

    @property
    def usage_input_key(self) -> str:
        """usage 中输入令牌的字段名。"""
        ...

    @property
    def usage_output_key(self) -> str:
        """usage 中输出令牌的字段名。"""
        ...


# ═══════════════════════════════════════════════════════════
#  通用请求/响应处理器
# ═══════════════════════════════════════════════════════════


def _dbg(message: str) -> None:
    if settings.get_debug_mode() in ('simple', 'verbose'):
        logger.info('[通用调试] %s', message)


def extract_responses_usage(event_data: JsonDict) -> JsonDict | None:
    """从原生 Responses 事件中提取 usage（公共辅助）。"""
    if not isinstance(event_data, dict):
        return None
    usage = event_data.get('usage')
    if isinstance(usage, dict):
        return usage
    response_obj = event_data.get('response')
    if isinstance(response_obj, dict):
        nested_usage = response_obj.get('usage')
        if isinstance(nested_usage, dict):
            return nested_usage
    return None


def handle_non_stream(
    ctx: Any,
    outbound: OutboundTransformer,
    client_fmt: ClientFormatter,
    payload: JsonDict,
    turn: JsonDict | None,
) -> Response:
    """通用非流式处理器。

    替代 chat.py 和 responses.py 中的 8 个 _handle_xxx_non_stream 函数。
    """
    from routes.common import apply_body_modifications, apply_header_modifications, log_usage

    upstream_payload = outbound.build_request(payload)
    url = outbound.build_url(ctx)
    headers = outbound.build_headers(ctx)
    upstream_payload = apply_body_modifications(upstream_payload, ctx.body_modifications)
    headers = apply_header_modifications(headers, ctx.header_modifications)

    upstream_payload['stream'] = False
    attach_upstream_request(turn, upstream_payload, headers)
    resp, err = forward_request(url, headers, upstream_payload)
    if err:
        attach_error(turn, {'stage': 'forward_request', 'message': 'upstream request failed'})
        finalize_turn(turn)
        return err

    raw = resp.json()
    attach_upstream_response(turn, raw)
    _dbg('上游原始响应=' + json.dumps(raw, ensure_ascii=False, default=str)[:1000])

    cc_response = outbound.parse_response(raw)
    result = client_fmt.format_response(cc_response, ctx.client_model)

    _dbg('格式化后响应=' + json.dumps(result, ensure_ascii=False, default=str)[:1000])
    usage_data = result.get('usage', {})
    log_usage('通用', usage_data, input_key=client_fmt.usage_input_key, output_key=client_fmt.usage_output_key)
    usage_tracker.record(
        ctx.client_model,
        usage_data,
        input_key=client_fmt.usage_input_key,
        output_key=client_fmt.usage_output_key,
    )
    attach_client_response(turn, result)
    finalize_turn(turn, usage=usage_data)
    return jsonify(result)


def handle_stream(
    ctx: Any,
    outbound: OutboundTransformer,
    client_fmt: ClientFormatter,
    payload: JsonDict,
    turn: JsonDict | None,
) -> Response:
    """通用流式处理器。

    替代 chat.py 和 responses.py 中的 8 个 _handle_xxx_stream 函数。
    """
    from routes.common import apply_body_modifications, apply_header_modifications

    upstream_payload = outbound.build_request(payload)
    url = outbound.build_url(ctx)
    headers = outbound.build_headers(ctx)
    upstream_payload = apply_body_modifications(upstream_payload, ctx.body_modifications)
    headers = apply_header_modifications(headers, ctx.header_modifications)

    upstream_payload['stream'] = True
    processor = outbound.create_stream_processor()

    def generate():
        for start_evt in client_fmt.start_events():
            yield start_evt

        attach_upstream_request(turn, upstream_payload, headers)
        resp, err = forward_request(url, headers, upstream_payload, stream=True)
        if err:
            attach_error(turn, {'stage': 'forward_request', 'message': str(err)})
            set_stream_summary(turn, {'status': 'error'})
            finalize_turn(turn)
            yield client_fmt.format_error(str(err))
            return

        event_count = 0
        client_items: list[str] = []
        last_usage: JsonDict | None = None

        for event in processor.iter_events(resp):
            append_upstream_event(turn, {'type': 'upstream_event', 'data': event})

            extracted = processor.extract_usage(event)
            if extracted is not None:
                last_usage = extracted

            if event_count < 10:
                _dbg(
                    f'上游事件#{event_count}='
                    + json.dumps(event, ensure_ascii=False, default=str)[:500]
                )

            for chunk in processor.process_event(event):
                if isinstance(chunk, dict):
                    chunk['model'] = ctx.client_model
                wrapped = client_fmt.wrap_stream_item(chunk)
                client_items.append(wrapped)
                append_client_event(turn, {'type': 'stream_item', 'data': chunk})
                if event_count < 10:
                    _dbg(
                        f'返回片段#{event_count}='
                        + json.dumps(chunk, ensure_ascii=False, default=str)[:500]
                    )
                yield wrapped

            event_count += 1

        for chunk in processor.finalize():
            if isinstance(chunk, dict):
                chunk['model'] = ctx.client_model
            wrapped = client_fmt.wrap_stream_item(chunk)
            client_items.append(wrapped)
            append_client_event(turn, {'type': 'stream_item', 'data': chunk})
            yield wrapped

        done = client_fmt.format_done()
        if done:
            append_client_event(turn, {'type': 'done'})
            yield done

        _dbg(f'流式响应结束，共 {event_count} 个事件')
        usage_tracker.record(
            ctx.client_model,
            last_usage,
            input_key=client_fmt.usage_input_key,
            output_key=client_fmt.usage_output_key,
        )
        set_stream_summary(turn, {
            'event_count': event_count,
            'client_item_count': len(client_items),
            'usage': last_usage,
        })
        attach_client_response(turn, {
            'type': 'stream.summary',
            'model': ctx.client_model,
            'event_count': len(client_items),
            'usage': last_usage,
        })
        finalize_turn(turn, usage=last_usage)

    return sse_response(generate())
