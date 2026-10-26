"""Task taxonomy and evidence requirements; no game answers or model routing.

Patterns describe information needs. They never establish that the KB has an
answer, nor encode hero facts, medal thresholds or season reward numbers.
"""
from __future__ import annotations
import re

TOPICS=[
 ('rating','对局表现评级',r'顶级分路|单局评分|对局评分|评牌|表现评级|(?:金牌|银牌|铜牌).*(?:排位|评分|表现)|(?:排位|对局|结算).*(?:金牌|银牌|铜牌|评分)','对局表现评级 金牌 银牌 顶级分路 排位积分'),
 ('honor','战力与荣誉',r'荣耀战力|荣耀称号|国服|国标|省标|市标|区标|分路段位|英雄战力|战力加成','荣耀战力 荣耀称号 分路段位 结算规则'),
 ('ranked','排位结算',r'排位|赛季|段位|加星|掉星|段位继承|升段|掉段|保星|排位积分|勇者积分','排位积分 加星 结算 赛季更新'),
 ('peak','巅峰赛',r'巅峰赛|巅峰积分|巅峰分|巅峰排名','巅峰赛 积分 参与条件 赛季规则'),
 ('matchmaking','匹配与选将',r'匹配机制|匹配算法|组队限制|五排|三排|双排|单排|组排|跨段|补位|预选|禁用|(?:排位|匹配).{0,8}(?:双排|三排|五排)','匹配 组队 选将 禁用 规则'),
 ('activity','活动与获取',r'活动|领奖|领取|兑换|返场|战令|限时|限免|免费.{0,10}皮肤|皮肤.{0,8}(?:怎么获得|如何获得)','活动 公告 获取条件 时间 限制'),
 ('mode','游戏模式',r'10[Vv]10|无限乱斗|火焰山|克隆大作战|大乱斗|训练营|娱乐模式|人机','模式 开放 玩法 规则'),
 ('economy','货币与资源',r'金币|点券|钻石|碎片|战令币|荣耀积分','货币 获取 上限 兑换 规则'),
 ('map','地图与资源',r'野怪|[红蓝][Bb][Uu][Ff][Ff]|暴君|主宰|兵线|防御塔|水晶|补刀|经验|复活|泉水|刷新|野区','峡谷 地图 资源 刷新 机制'),
 ('combat','战斗机制',r'韧性|护甲|法抗|冷却缩减|吸血|穿透|暴击|攻速|移速|真实伤害|法术伤害|物理伤害|免伤|控制|伤害','战斗机制 属性 上限 计算'),
 ('equipment','装备与铭文',r'装备|出装|铭文|(?:购买|出售).{0,5}(?:价格|金币)','装备 铭文 属性 价格'),
 ('summoner','召唤师技能',r'召唤师|闪现|惩击|净化|狂暴|疾跑','召唤师技能 效果 冷却'),
 ('social','社交与战队',r'好友|亲密度|师徒|战队|小队|赠送|赠礼|观战|回放|战报','好友 亲密度 战队 限制 规则'),
 ('platform','平台与互通',r'安卓|iOS|苹果|微信区|QQ区|跨平台|转区|换区|互通|数据迁移','安卓 苹果 微信 QQ 互通 官方规则'),
 ('esports','赛事信息',r'KPL|比赛赛程|赛事|冠军|比赛结果','赛事 官方 赛程 结果'),
]
RULE_TOPICS={'rating','honor','ranked','peak','matchmaking','platform','esports'}
VOLATILE_PREDICATE=re.compile(r'多少|几[秒分钟颗星分次天个款]|多久|上限|门槛|条件|限制|获得|奖励|结算|积分|刷新|改动|数值|价格|售价|开放|可用|能不能|能否|可以|会不会|什么时候')
HISTORY=re.compile(r'历史|旧版|当年|当时|以前|原先|S\d{1,2}赛季|20\d{2}年|只按.*快照|演示|ACT-\d|KB-\d',re.I)
CURRENT=re.compile(r'最新|当前|现在|本周|今天|本赛季|现行|实时')
DEFINITION=re.compile(r'是什么|什么意思|什么是|含义|概念|术语')
COMPONENTS=re.compile(r'顶级分路|单局评分|对局评分|金牌|银牌|铜牌|排位积分|勇者积分|荣耀战力|荣耀称号|段位继承|加星|掉星|胜利|失败|赢|输')
CONCEPT_ALIASES={'单局评分':('单局评分','对局评分','评分机制','结算评价','评价体系'),
 '对局评分':('单局评分','对局评分','评分机制','结算评价','评价体系'),
 '金牌':('金牌','金银牌'),'银牌':('银牌','金银牌'),
 '荣耀战力':('荣耀战力','英雄战力'),'顶级分路':('顶级分路',)}

def concept_matches(concept,text):return any(term in text for term in CONCEPT_ALIASES.get(concept,(concept,)))
GENERIC_KEYWORDS={'排位','段位','等级','经验','金币','分路','野区','地图','装备','英雄','技能','胜负','皮肤','价格'}
SYSTEM_SUBJECTS=['顶级分路','单局评分','对局评分','荣耀战力','荣耀称号','分路段位','排位赛','排位积分','勇者积分','巅峰赛','匹配机制','段位继承','亲密度','战队','小队']

def question_policy(question):
    found=[(key,label,terms) for key,label,pattern,terms in TOPICS if re.search(pattern,question,re.I)]
    keys=[key for key,_,_ in found]
    historical=bool(HISTORY.search(question)) and not CURRENT.search(question)
    concept_only=bool(DEFINITION.search(question) and re.search(r'区别|对比|比较|异同',question)
        and not re.search(r'多少|几[分颗星秒次]|数值|公式|计算|阈值|门槛|上限|加星|掉星|奖励|标准|条件',question))
    definition=bool(DEFINITION.search(question)) and not VOLATILE_PREDICATE.search(question)
    volatile=not historical and bool(set(keys)&RULE_TOPICS and not definition
        or 'rating' in keys or 'activity' in keys and not DEFINITION.search(question)
        or set(keys)&{'mode','economy','map','combat','equipment','social'} and VOLATILE_PREDICATE.search(question))
    if concept_only and not CURRENT.search(question):volatile=False
    # A naked medal comparison can mean match ratings or regional honor badges.
    medals=re.search('金牌|银牌|铜牌',question)
    ambiguous=bool(medals and not re.search(r'排位|对局|评分|评牌|表现|结算|顶级分路|荣耀|战力|国标|省标|市标|区标|比赛|赛事|奥运',question))
    return {'topics':keys,'topic_labels':[label for _,label,_ in found],'version_sensitive':volatile,'concept_only':concept_only,
      'historical_scope':historical,'ambiguous_medal_system':ambiguous,
      'components':list(dict.fromkeys(COMPONENTS.findall(question))),
      'search_terms':[terms for _,_,terms in found[:2]]}

def local_confidence(question,sources,plan):
    """Retrieval support score, not a calibrated probability or a fact verdict."""
    if not sources:return 0.
    if plan.get('entities'):return .85
    if any(str(s.get('doc_id','')).startswith(('ITEM-','SUMMONER-')) and s.get('source_authority')=='official' for s in sources):return .85
    matches=[s.get('retrieval_match',{}) for s in sources]
    if matches and all(m.get('kind')=='specific_term' for m in matches):return .85
    policy=plan.get('evidence_policy') or question_policy(question)
    body='\n'.join(s.get('content','') for s in sources)
    components=[x for x in policy['components'] if x not in {'赢','输','胜利','失败'}]
    if components and not all(concept_matches(x,body) for x in components):return .25
    if policy['version_sensitive']:return .4
    return .45 if any(m.get('kind')=='broad_keyword' for m in matches) else .75

def topical_source(question,doc,matched):
    """Prevent a generic introduction from masquerading as a specialized rule."""
    policy=question_policy(question);body=doc.get('content','')
    if policy['topics'] and policy['topics'][0] in RULE_TOPICS:
        components=[x for x in policy['components'] if len(x)>=2 and x not in {'胜利','失败'}]
        if components and not any(concept_matches(x,body) for x in components):return False
        if set(matched)<=GENERIC_KEYWORDS and doc.get('doc_id') in {'BASIC-INTRO','BASIC-MAP','BASIC-RULES'}:
            return False
    return True

def search_query(question,entities,attempt=0,missing=()):
    """Preserve scope, replace a failed query by the missing evidence need."""
    cleaned=re.sub(r'https?://\S+|请联网核实|请联网|联网查询|帮我查一下|请问','',question).strip()
    policy=question_policy(cleaned)
    anchors=[e['cname'] for e in entities]
    anchors += [skin.split('&')[0] for e in entities for skin in str(e.get('skin_name','')).split('|') if len(skin.split('&')[0])>=2 and skin.split('&')[0] in cleaned]
    anchors+=re.findall(r'20\d{2}年?|S\d{1,2}|\d+周年|周年|免费|不花钱|不用.{0,8}(?:碎片|点券)|不使用.{0,8}(?:碎片|点券)|兑换截止|领取时间|入口|截止|任务|[一二三四]技能|大招|《[^》]+》|“[^”]+”',cleaned)
    anchors+=policy['components']
    anchors=list(dict.fromkeys(anchors))
    if attempt and policy['search_terms']:
        focus=' '.join(str(x)[:50] for x in missing[:2]) if missing else policy['search_terms'][0]
        body=' '.join([*anchors,cleaned,focus,'正式服 更新公告' if policy['version_sensitive'] else '规则 说明'])
    else:
        body=cleaned
        if policy['version_sensitive']:body+=' 正式服 规则 版本更新'
    suffix=' site:pvp.qq.com' if not attempt else ' 官方'
    prefix='王者荣耀 '
    budget=128-len(prefix)-len(suffix)
    if len(body)>budget:
        protected=' '.join(anchors)
        body=(protected+' '+body)[:budget]
    return prefix+body[:budget]+suffix

def private_profile_query(question):
    if not re.search(r'我的|我(?:目前|现在|已经|当前)?(?:的)?(?:段位|战力|胜率|巅峰分|信誉分)',question):return False
    return bool(re.search(r'战力|胜率|段位|巅峰分|信誉分|对局记录',question) and re.search(r'多少|几[分颗星]|什么|查询|查一下|查查|查看|读取',question)
                and not re.search(r'怎么|如何|为什么|规则|含义|意思|提升|计算|不涨',question))
