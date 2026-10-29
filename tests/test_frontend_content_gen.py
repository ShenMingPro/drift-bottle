"""直接执行页面真实 JavaScript，验证刷新推荐及 SSE/回退行为。"""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


INDEX_HTML = Path(__file__).parents[1] / "public" / "index.html"
NODE = shutil.which("node")

HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const page = fs.readFileSync(process.argv[1], 'utf8');
const scenario = process.argv[2];
const parserStart = page.indexOf('function createSseParser(');
const parserEnd = page.indexOf('const { createApp,', parserStart);
const stateStart = page.indexOf('const contentSuggestions = ref([]);');
const stateEnd = page.indexOf('const pickLoading = ref(false);', stateStart);
const functionsStart = page.indexOf('async function loadContentSuggestions()');
const functionsEnd = page.indexOf('async function onWriteConfirm()', functionsStart);
const calls = [], toasts = [], queue = [];
const context = {
  ref: value => ({value}),
  userInfo: {userId: 7, nickname: '测试'}, currentPage: {value: 'home'},
  writeContent: {value: ''},
  localStorage: {getItem: () => 'test-token', removeItem: () => {}},
  vant: {showToast: message => toasts.push(message)},
  URLSearchParams, TextDecoder, Date,
};
async function fetch(url, init = {}) {
  const parsed = new URL(url, 'http://local.test');
  const body = init.body ? JSON.parse(init.body) : {};
  calls.push({path: parsed.pathname, method: init.method || 'GET',
    userId: String(body.userId ?? parsed.searchParams.get('userId')),
    previous: body.previousSuggestions || parsed.searchParams.getAll('previousSuggestions'),
    body: init.body || '', contentType: init.headers?.['Content-Type']});
  if (!queue.length) throw new Error('测试响应队列已空');
  const response = queue.shift();
  if (response instanceof Error) throw response;
  return typeof response === 'function' ? await response() : response;
}
context.fetch = fetch;
context.authFetch = fetch;
vm.runInNewContext(
  page.slice(parserStart, parserEnd) + '\n' +
  page.slice(stateStart, stateEnd) + '\n' +
  page.slice(functionsStart, functionsEnd) + '\n' +
  `globalThis.api = {
    load: loadContentSuggestions,
    get suggestions() { return contentSuggestions.value; },
    get loading() { return contentGenLoading.value; }
  };`, context);
const api = context.api;
const batch = number => [1, 2, 3].map(index => ({
  style: '聊天', content: `第${number}批建议${index}`, proof: `proof-${number}-${index}`,
}));
const event = (name, data) => `event: ${name}\r\ndata: ${JSON.stringify(data)}\r\n\r\n`;
function stream(events, split = false) {
  const body = events.map(([name, data]) => event(name, data)).join('');
  const bytes = new TextEncoder().encode(body);
  return new Response(new ReadableStream({start(controller) {
    // 逐字节拆分还会切开中文 UTF-8，确保解码器和解析器均能保留状态。
    if (split) for (let i = 0; i < bytes.length; i++) controller.enqueue(bytes.slice(i, i + 1));
    else controller.enqueue(bytes);
    controller.close();
  }}), {headers: {'Content-Type': 'text/event-stream'}});
}
const success = (number, split = false) => stream([
  ['result', {suggestions: batch(number)}], ['done', {}],
], split);
const jsonResponse = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: {'Content-Type': 'application/json'},
});
const tick = () => new Promise(resolve => setImmediate(resolve));
async function run() {
  let extra = {};
  if (scenario === 'history') {
    for (let number = 1; number <= 13; number++) {
      queue.push(success(number));
      await api.load();
    }
  } else if (scenario === 'split') {
    queue.push(success(1, true));
    await api.load();
  } else if (scenario === 'guard') {
    let release;
    queue.push(() => new Promise(resolve => { release = resolve; }));
    const first = api.load();
    const second = api.load();
    release(success(1));
    await Promise.all([first, second]);
  } else if (scenario === 'fallback') {
    queue.push(success(1));
    await api.load();
    let release;
    queue.push(new Error('流式连接中断'));
    queue.push(() => new Promise(resolve => { release = resolve; }));
    let completed = false;
    const pending = api.load().then(() => { completed = true; });
    await tick();
    extra.completedBeforeFallback = completed;
    extra.loadingDuringFallback = api.loading;
    release(jsonResponse({suggestions: batch(2)}));
    await pending;
    await tick();
  } else if (scenario === 'missing_result') {
    queue.push(stream([['progress', {stage: 'generating'}], ['done', {}]]));
    queue.push(jsonResponse({suggestions: batch(2)}));
    await api.load();
  } else if (scenario === 'error_event') {
    queue.push(stream([['error', {message: '暂时未生成不同的内容，请重试'}], ['done', {}]]));
    await api.load();
  } else if (scenario === 'http_error') {
    queue.push(jsonResponse({detail: 'AI服务暂时不可用'}, 503));
    await api.load();
  } else if (scenario === 'fallback_failure') {
    queue.push(new Error('流式连接中断'));
    queue.push(jsonResponse({detail: '请稍后重试'}, 503));
    await api.load();
    await tick();
  } else if (scenario === 'invalid_result') {
    queue.push(stream([['result', {suggestions: [{content: 'x'.repeat(201)}]}], ['done', {}]]));
    queue.push(jsonResponse({suggestions: []}));
    await api.load();
    await tick();
  } else if (scenario === 'user_history') {
    queue.push(success(1), success(2), success(3));
    await api.load();
    context.userInfo.userId = 8;
    await api.load();
    context.userInfo.userId = 7;
    await api.load();
  } else if (scenario === 'stale_user') {
    let release;
    queue.push(() => new Promise(resolve => { release = resolve; }));
    const pending = api.load();
    context.userInfo.userId = 8;
    release(success(1));
    await pending;
    extra.staleSuggestions = JSON.parse(JSON.stringify(api.suggestions));
    queue.push(success(2));
    await api.load();
  } else if (scenario === 'stale_fallback_body') {
    let release;
    queue.push(new Error('流式连接中断'));
    queue.push({ok: true, json: () => new Promise(resolve => { release = resolve; })});
    const pending = api.load();
    await tick();
    context.userInfo.userId = 8;
    release({suggestions: batch(1)});
    await pending;
    extra.staleSuggestions = JSON.parse(JSON.stringify(api.suggestions));
  } else if (scenario === 'repeated_result') {
    queue.push(success(1));
    await api.load();
    queue.push(success(1), jsonResponse({suggestions: batch(1)}));
    await api.load();
  } else throw new Error('未知场景');
  return {calls, toasts, suggestions: api.suggestions, loading: api.loading, ...extra};
}
run().then(result => console.log(JSON.stringify(result))).catch(error => {
  console.error(error.stack); process.exitCode = 1;
});
"""


def run_frontend(scenario):
    if not NODE:
        pytest.skip("前端行为测试需要 Node.js")
    result = subprocess.run(
        [NODE, "-e", HARNESS, str(INDEX_HTML), scenario],
        capture_output=True, text=True, encoding="utf-8", timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def contents(number):
    return [f"第{number}批建议{index}" for index in range(1, 4)]


def test_refresh_passes_previous_content_and_keeps_only_last_30():
    result = run_frontend("history")
    assert result["calls"][0]["previous"] == []
    assert result["calls"][1]["previous"] == contents(1)
    assert result["calls"][-1]["previous"] == [
        text for number in range(3, 13) for text in contents(number)
    ]
    assert all("proof-" not in call["body"] for call in result["calls"])
    assert all(call["method"] == "POST" for call in result["calls"])
    assert all(call["contentType"] == "application/json" for call in result["calls"])
    assert [item["content"] for item in result["suggestions"]] == contents(13)


def test_sse_event_data_and_utf8_can_span_chunks():
    result = run_frontend("split")
    assert [item["content"] for item in result["suggestions"]] == contents(1)
    assert len(result["calls"]) == 1
    assert result["loading"] is False


def test_loading_blocks_duplicate_clicks():
    result = run_frontend("guard")
    assert len(result["calls"]) == 1
    assert [item["content"] for item in result["suggestions"]] == contents(1)


def test_fallback_awaits_result_and_reuses_same_previous_content():
    result = run_frontend("fallback")
    assert result["completedBeforeFallback"] is False
    assert result["loadingDuringFallback"] is True
    assert result["calls"][1]["previous"] == contents(1)
    assert result["calls"][2]["previous"] == contents(1)
    assert result["calls"][1]["body"] == result["calls"][2]["body"]
    assert result["calls"][1]["method"] == result["calls"][2]["method"] == "POST"
    assert [item["content"] for item in result["suggestions"]] == contents(2)
    assert result["loading"] is False


def test_missing_result_uses_fallback():
    result = run_frontend("missing_result")
    assert len(result["calls"]) == 2
    assert [item["content"] for item in result["suggestions"]] == contents(2)


@pytest.mark.parametrize("scenario", ["error_event", "http_error", "fallback_failure", "invalid_result"])
def test_generation_failure_shows_error_without_fake_suggestions(scenario):
    result = run_frontend(scenario)
    assert result["suggestions"] == []
    assert result["toasts"]
    assert result["loading"] is False
    assert len(result["calls"]) == (2 if scenario in {"fallback_failure", "invalid_result"} else 1)


def test_previous_content_is_scoped_to_user():
    result = run_frontend("user_history")
    assert result["calls"][1]["previous"] == []
    assert result["calls"][2]["previous"] == contents(1)
    assert [call["userId"] for call in result["calls"]] == ["7", "8", "7"]


def test_user_switch_discards_previous_users_inflight_response():
    result = run_frontend("stale_user")
    assert result["staleSuggestions"] == []
    assert result["calls"][1]["previous"] == []
    assert [item["content"] for item in result["suggestions"]] == contents(2)


def test_user_switch_while_reading_fallback_body_discards_response():
    result = run_frontend("stale_fallback_body")
    assert result["staleSuggestions"] == []
    assert result["loading"] is False


def test_repeated_server_result_is_not_shown_as_new_content():
    result = run_frontend("repeated_result")
    assert result["suggestions"] == []
    assert result["toasts"]
    assert result["calls"][1]["previous"] == contents(1)
    assert result["calls"][2]["previous"] == contents(1)
