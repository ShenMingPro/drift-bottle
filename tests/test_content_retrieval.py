"""内容生成前的相关话题检索测试。"""

import asyncio
import json

import app.agents.content_gen_agent as content_gen


def test_fetch_user_info_ranks_topics_by_interest(monkeypatch):
    async def profile(_user_id):
        return json.dumps({
            "爱好": "摄影,跑步",
            "学校": "测试大学",
            "城市": "测试市",
        }, ensure_ascii=False)

    async def topics():
        return json.dumps({
            "热门话题": ["篮球比赛", "摄影展", "食堂吐槽"],
        }, ensure_ascii=False)

    monkeypatch.setattr(content_gen, "_query_user_profile", profile)
    monkeypatch.setattr(content_gen, "_query_recent_hot_topics", topics)

    result = asyncio.run(content_gen.fetch_user_info({"user_id": 7}))
    topics_data = json.loads(result["hot_topics"])

    assert topics_data["相关话题"][0] == "摄影展"
    assert "篮球比赛" in topics_data["相关话题"]
