# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest

from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from tests.unit.conftest import make_run_from_cli

ROW_HEADERS = {"x-session-token": "abc", "Host": "pool.example"}


def test_mooncake_trace_accepts_headers():
    t = MooncakeTrace(input_length=10, headers=ROW_HEADERS)
    assert t.headers == ROW_HEADERS


def test_mooncake_trace_headers_default_to_none():
    t = MooncakeTrace(input_length=10)
    assert t.headers is None


@pytest.mark.parametrize(
    "row",
    [
        pytest.param({"input_length": 100, "output_length": 10}, id="input_length"),
        pytest.param({"text_input": "Hello", "output_length": 10}, id="text_input"),
        pytest.param(
            {"messages": [{"role": "user", "content": "Hello"}], "output_length": 10},
            id="messages",
        ),
        pytest.param({"payload": {"prompt": "Hello", "max_tokens": 10}}, id="payload"),
    ],
)
class TestMooncakeTraceHeadersReachTurn:
    def test_row_headers_become_turn_extra_headers(
        self, row, create_jsonl_file, default_cfg, mock_prompt_generator
    ):
        filename = create_jsonl_file(
            [orjson.dumps({"timestamp": 0, **row, "headers": ROW_HEADERS}).decode()]
        )
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            run=make_run_from_cli(default_cfg),
            prompt_generator=mock_prompt_generator,
        )

        conversations = loader.convert_to_conversations(loader.load_dataset())

        assert conversations[0].turns[0].extra_headers == ROW_HEADERS

    def test_row_without_headers_leaves_turn_extra_headers_unset(
        self, row, create_jsonl_file, default_cfg, mock_prompt_generator
    ):
        filename = create_jsonl_file([orjson.dumps({"timestamp": 0, **row}).decode()])
        loader = MooncakeTraceDatasetLoader(
            filename=filename,
            run=make_run_from_cli(default_cfg),
            prompt_generator=mock_prompt_generator,
        )

        conversations = loader.convert_to_conversations(loader.load_dataset())

        assert conversations[0].turns[0].extra_headers is None
