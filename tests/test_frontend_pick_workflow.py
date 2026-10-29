"""页面接入可恢复拾取工作流的契约与运行时测试。"""

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
const start = page.indexOf('function pickResultFromWorkflow(');
const end = page.indexOf('async function openHistory()', start);
const source = page.slice(start, end);
const calls = [], toasts = [], queue = [];
const chatTarget = {};
const context = {
  ref: value => ({ value }), reactive: value => value,
  userInfo: { userId: 7 }, pickTarget: { university: '测试大学', city: '测试市', province: '测试省' },
  pickLoading: { value: false }, pickChances: { value: 1 }, pickTimer: null,
  pickResult: { value: null }, showPickResult: { value: false },
  showSmartPickApproval: { value: false }, smartPickRun: { value: null },
  smartPickHints: { value: [] }, smartPickSelectedHint: { value: '' }, smartPickLoading: { value: false },
  chatTarget, chatMessages: { value: [] }, chatInput: { value: '' }, currentPage: { value: 'main' },
  vant: { showToast: value => toasts.push(value), closeToast: () => {} },
  startPickCountdown: () => {}, syncChances: () => {}, encodeURIComponent, URLSearchParams,
};
context.loadMessages = async () => { context.chatInput.value = ''; };
async function authFetch(url, init = {}) {
  calls.push({ url, body: init.body ? JSON.parse(init.body) : null, method: init.method || 'GET' });
  if (!queue.length) throw new Error('empty queue');
  const response = queue.shift();
  return { ok: response.ok !== false, json: async () => response.body };
}
context.authFetch = authFetch;
vm.runInNewContext(source + `
globalThis.api = { start: pickBottleSmart, approve: approveSmartPick,
  get approval() { return showSmartPickApproval.value; }, get hints() { return smartPickHints.value; },
  get run() { return smartPickRun.value; }, get result() { return pickResult.value; },
  get shown() { return showPickResult.value; }, get selected() { return smartPickSelectedHint.value; },
  get page() { return currentPage.value; }, get input() { return chatInput.value; },
  get chatTarget() { return chatTarget; } };
`, context);
const api = context.api;
const match = { status: 'success', match_level: 'university', match_reason: '同校', best_match: { bottle_id: 12, user_id: 23, content: '喜欢摄影', nickname: '小林' } };
const awaiting = { runId: 'run-1', status: 'awaiting_approval', match, approvedHints: ['安全建议'], iceBreak: { hints: ['安全建议'] } };
const approved = { runId: 'run-1', status: 'approved', match, approvedHints: ['安全建议'], pickRecorded: true };
async function run() {
  queue.push({ body: awaiting });
  await api.start();
  const afterStart = { approval: api.approval, hints: api.hints, run: api.run && api.run.runId, selected: api.selected };
  queue.push({ body: approved });
  queue.push({ body: { id: 9, targetUserId: 23, targetNickname: '小林', targetGender: '女', targetAge: 21 } });
  await api.approve();
  console.log(JSON.stringify({
    afterStart,
    afterApprove: { result: api.result, shown: api.shown, approval: api.approval, page: api.page,
      input: api.input, chatTarget: api.chatTarget, calls },
    toasts
  }));
}
run().catch(error => { console.error(error.stack); process.exitCode = 1; });
"""


def page_source() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


def smart_pick_source() -> str:
    page = page_source()
    start = page.index("async function pickBottleSmart()")
    end = page.index("async function openHistory()", start)
    return page[start:end]


def test_page_exposes_recoverable_workflow_entrypoint():
    page = page_source()

    assert "onSmartPickClick" in page
    assert "showSmartPickApproval" in page
    assert "smartPickSelectedHint" in page
    assert "智能全流程" in page


def test_smart_pick_uses_run_and_resume_endpoints():
    source = smart_pick_source()

    assert "authFetch('/api/pick/runs'" in source
    assert "/api/pick/runs/${encodeURIComponent(run.runId)}/resume" in source
    assert "approved: true" in source
    assert "approved: false" in source
    assert "approvedHints" in source


def test_smart_pick_only_surfaces_approved_hints():
    page = page_source()

    assert "data.approvedHints" in page
    assert "smartPickHints.value = Array.isArray(data.approvedHints)" in page
    assert "smartPickRun.match.best_match.content" in page


def test_smart_pick_opens_chat_and_only_prefills_input():
    source = page_source()

    assert "async function openChatForWorkflow(data, hint)" in source
    assert "chatInput.value = hint || ''" in source
    assert "currentPage.value = 'chatDetail'" in source
    workflow_source = source[
        source.index("async function openChatForWorkflow("):
        source.index("async function openHistory(", source.index("async function openChatForWorkflow("))
    ]
    assert "sendChatMessage()" not in workflow_source


def test_client_reads_server_owned_chances_instead_of_writing_them():
    page = page_source()

    assert "/api/update-chances" not in page
    assert "useBonus: isBonusWrite.value" in page
    assert "authFetch(`/api/user-chances?userId=${userInfo.userId}`)" in page


def test_smart_pick_runtime_pauses_then_resumes():
    if not NODE:
        pytest.skip("前端行为测试需要 Node.js")
    result = subprocess.run(
        [NODE, "-e", HARNESS, str(INDEX_HTML)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["afterStart"]["approval"] is True
    assert payload["afterStart"]["hints"] == ["安全建议"]
    assert payload["afterStart"]["selected"] == "安全建议"
    assert payload["afterApprove"]["result"]["bottleUserId"] == 23
    assert payload["afterApprove"]["shown"] is False
    assert payload["afterApprove"]["approval"] is False
    assert payload["afterApprove"]["page"] == "chatDetail"
    assert payload["afterApprove"]["input"] == "安全建议"
    assert payload["afterApprove"]["chatTarget"]["id"] == 23
    assert payload["afterApprove"]["chatTarget"]["conversationId"] == 9
    assert payload["afterApprove"]["calls"][0]["url"] == "/api/pick/runs"
    assert payload["afterApprove"]["calls"][1]["url"] == "/api/pick/runs/run-1/resume"
    assert payload["afterApprove"]["calls"][2]["url"] == "/api/conversations"


def test_smart_pick_does_not_open_chat_before_business_commit():
    if not NODE:
        pytest.skip("前端行为测试需要 Node.js")
    pending = HARNESS.replace("pickRecorded: true", "pickRecorded: false")
    result = subprocess.run(
        [NODE, "-e", pending, str(INDEX_HTML)],
        capture_output=True, text=True, encoding="utf-8", timeout=15,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["afterApprove"]["page"] == "main"
    assert payload["afterApprove"]["approval"] is True
    assert len(payload["afterApprove"]["calls"]) == 2
