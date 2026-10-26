"""Classify tasks, never guess corpus coverage with an LLM router."""
from __future__ import annotations
from datetime import date, datetime
from zoneinfo import ZoneInfo
import re

from .official import OfficialStore
from .conversation import contextualize
from .question_policy import question_policy

FRESH = re.compile(r"最新|本周|本月|本赛季|新赛季|新版本|当前|现在|实时|联网|核实|查询官网|官网现有|今天|刷新时间|多久刷新")
SUMMARY = re.compile(r"总结|概括|汇总|有哪些技能|技能介绍|介绍一下|介绍下|介绍")
COMPARE = re.compile(r"比较|对比|区别|异同|分别|共同|谁更|哪个.{0,8}更|哪位.{0,8}更")
AGGREGATE = re.compile(r"一共|总数|数量|条目数|个数|英雄数|皮肤数|装备数|收录数|多少[个位种款项]?|几[个位种款项名人]")


def plan_question(question: str, store: OfficialStore | None, history=None, *, client=None, trace=None) -> dict:
    original = question
    context = contextualize(question, history, store, client=client, trace=trace)
    question = context['question']
    entities = store.entities(question) if store else []
    task = "fact"
    aggregate_subject = re.search(r"皮肤|英雄|装备|目录|职业|射手|战士|坦克|法师|辅助|刺客|召唤师技能", question)
    skill_measure = re.search(r"技能|被动|冷却|伤害|击飞|眩晕|减速|消耗|秒|总价|售价|价格|金币|加成|属性|效果|暴击|攻速|移速|攻击|防御|生命|法术|物理|吸血|穿透|护甲|数值|倍率|等级|段位|上线|返利|概率|点券|刷新|如何|为什么|作用|机制|price", question, re.I)
    from .statistics import supported_statistics
    summoner_count=bool(re.search('召唤师技能',question) and AGGREGATE.search(question) and not re.search('伤害|冷却|消耗|解锁|作用|机制|效果|英雄|皮肤|装备',question))
    if AGGREGATE.search(question) and aggregate_subject and (not skill_measure or summoner_count) and supported_statistics(question):
        task = "aggregate"
    elif COMPARE.search(re.sub(r'(?:不需要|不用|不要|无需|不必|不)(?:比较|对比)','',question)):
        task = "compare"
    elif SUMMARY.search(question):
        task = "summarize"
    current_year=str(datetime.now(ZoneInfo('Asia/Shanghai')).year)
    year_live=bool(re.search(current_year+r'年?.*(?:活动|皮肤|版本|赛季)',question)) and not re.search('只按.*目录快照',question)
    fresh=bool(FRESH.search(question) or year_live)
    explicit_demo = bool(re.search(r'演示资料|ACT-\d|KB-\d|2025年.*规则|知识库.*(?:收录|历史|2025)',question) or
                         re.search(r'知识库.*活动',question) and not fresh)
    policy=question_policy(question)
    # Version-sensitive system rules need verification even without "latest".
    # Named hero fields keep the existing bounded-fresh snapshot fast path.
    item_scope=bool(store and any(row['item_name'] in question for row in store.items))
    cached_item_field=item_scope and not (set(policy['topics'])-{'equipment','economy','combat'})
    if not explicit_demo and not cached_item_field and policy['version_sensitive'] and (not entities or set(policy['topics'])&{'rating','ranked','honor','peak','matchmaking','activity','platform','esports'}):fresh=True
    unresolved = context['needs_clarification']
    # Universal quantifiers describe the whole roster, even when phrased as
    # "每一位" instead of "所有". A top-k sample cannot satisfy that scope.
    all_heroes = bool(re.search(r'(?:全(?:部|体)?|所有|每(?:一)?(?:位|名|个)?|各(?:位|名|个)?)(?:的)?\s*(?:战士|法师|坦克|刺客|射手|辅助)?英雄',question))
    from .conversation import known_subject
    missing_hero_scope=task in {'summarize','compare'} and not entities and not known_subject(question,store) and bool(re.search('英雄',question))
    catalog_only=bool(re.search('名单|列表|列出|有哪些',question)) and not re.search('技能|故事|出装|玩法|克制|伤害|控制',question)
    unbounded = (len(entities) > (12 if task=='summarize' else 6)
                 or all_heroes and not entities and task!='aggregate' and not catalog_only
                 or missing_hero_scope and not catalog_only
                 or bool(re.search(r'所有活动|全部活动',question)))
    from .reference import facets
    wanted=facets(question)
    if set(policy['topics'])&{'rating','honor','ranked','peak'}:wanted=[f for f in wanted if f!='lane']
    return {"task": task, "question": question, "original_question": original,"facets":wanted,"context":context,'evidence_policy':policy,
            "entities": entities, "fresh": fresh, "explicit_demo": explicit_demo,
            "expected_entities": [str(e["ename"]) for e in entities],
            "entity_names": [e["cname"] for e in entities], "needs_clarification": unresolved or unbounded or context['needs_clarification']}


def activity_valid(doc: dict, question: str) -> bool:
    if doc.get("doc_type") != "activity":
        return True
    # Explicit historical corpus questions keep historical rules available.
    if re.search(r'收录|历史|演示|202[0-5]',question) or '知识库' in question and not FRESH.search(question):
        return True
    period = str(doc.get("activity_time", ""))
    ends = re.findall(r"\d{4}-\d{2}-\d{2}", period)
    return not ends or date.fromisoformat(ends[-1]) >= datetime.now(ZoneInfo("Asia/Shanghai")).date()
