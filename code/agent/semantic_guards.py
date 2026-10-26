"""Bounded checks for scaling bases and condition strength, before LLM audit.

These reject explicit contradictions; they are not a general entailment engine.
No hero answers or per-question rules belong in this module.
"""
from __future__ import annotations
import re
from urllib.parse import urlsplit
from .transport import APIError
from .evidence import normalize_quote
from .question_policy import question_policy,concept_matches

def unsupported_absence_claim(claim):
    text=claim['text']
    metadata_absence=re.search(r'(?:官方|官网|公告|原文|文档|资料|来源|网页)[^。；]{0,45}(?:(?:未|没有|尚未)(?:明确|直接|完整|具体|单独|进一步)?(?:说明|提到|提及|给出|提供|解释|定义|证明)|无法确认|尚未确认)',text)
    if not metadata_absence:return False
    return not any(normalize_quote(text) in normalize_quote(q['quote']) for q in claim.get('quotes',[]))

def validate_comparison_scope(question,claims):
    expected=[x for x in question_policy(question)['components'] if x not in {'加星','掉星','胜利','失败','赢','输'}]
    if len(set(expected))<2:return
    mentioned={x for x in expected if any(concept_matches(x,c['text']) for c in claims)}
    for claim in claims:
        if re.search(r'两者|二者|前者|后者|区别在于|两种',claim['text']) and len(mentioned)<2:
            raise APIError('answer: 比较对象未覆盖用户所问概念，不能把子项之间的比较冒充原问题的比较')

def validate_concept_scope(question,claims):
    """Do not expand definitions into unasked, version-dependent scoring rules."""
    if not question_policy(question).get('concept_only'):return
    for claim in claims:
        if re.search(r'加分|扣分|加成|系数|停止增长|段位上限|计算公式|计分公式',claim['text']):
            raise APIError('answer: 用户只问概念与区别，请删除未问的加扣分、上限和计算细节，用完整原文支持各自的定义与范围')

_STAT = re.compile(r'(额外|最大|已损失|已损|当前|剩余|基础|总)(生命值?|法力值?|物理攻击力?|法术攻击力?|攻击力|法术强度|物理防御|法术防御)')
_UNLOCK = re.compile(r'解锁|解除.{0,5}(?:封印|限制)|释放前置|施放前提')
_NEGATED = re.compile(r'不是|并非|不等于|不受|不依赖|无需|并不|而非')
_ACTION = re.compile(r'普通攻击|普攻|技能命中|命中标记|击中标记|叠加标记|叠加剑气')
_NECESSARY = re.compile(r'只有|必须|需要|需先|方可|才能|前提|前置|必要|条件')


def hosted_guide(url):
    """Known content-hosting routes are not canonical hero reference pages."""
    try:
        parsed=urlsplit(url);host=(parsed.hostname or '').lower().rstrip('.')
        return (host=='pvp.qq.com' or host.endswith('.pvp.qq.com')) and parsed.path.lower() in {
            '/v/detail.shtml','/web201706/newsdetail.shtml'}
    except (TypeError,ValueError):return False


def reference_rank(source):
    if hosted_guide(source.get('url','')) or source.get('trust_level') in {
            'official_hosted_guide','trusted_secondary','historical_guide','test_server'}:
        return 0
    if source.get('trust_level')=='official_reference':return 2
    try:path=urlsplit(source.get('url','')).path
    except (TypeError,ValueError):return 0
    return int(source.get('trust_level')=='official_page' and '/herodetail/' in path)


def scaling_bases(text):
    """Only scaling assertions/formulas; an increase to max HP is not a coefficient."""
    result=set()
    for clause in re.split(r'[，。；;\n]',text):
        if _NEGATED.search(clause):continue
        for m in _STAT.finditer(clause):
            before=clause[max(0,m.start()-18):m.start()];after=clause[m.end():m.end()+12]
            coefficient=bool(re.search(r'\d+(?:\.\d+)?\s*[%％]\s*(?:自身|目标|其)?$',before))
            described=bool(re.match(r'(?:的)?(?:加成|系数|比例|百分比|百分之)',after) or
                           re.search(r'(?:基于|依据|取决于|根据|受到?|依赖|按)(?:自身|其|目标)?$',before))
            if not (coefficient or described):continue
            basis={'已损失':'已损','剩余':'当前'}.get(m[1],m[1]);stat=m[2].removesuffix('值')
            result.add((basis,stat))
    return result


def _quoted(claim,sources):
    by_id={s['id']:s for s in sources}
    return [(by_id[q['source_id']],q['quote']) for q in claim['quotes']]


def validate_scaling_bases(claims,sources):
    for claim in claims:
        requested=scaling_bases(claim['text'])
        if not requested:continue
        quoted=_quoted(claim,sources)
        best=max(reference_rank(s) for s,_ in quoted)
        proofs=[q for s,q in quoted if reference_rank(s)==best]
        # When a weaker source alone is cited, an available reference for the
        # same unambiguous entity must not be silently contradicted.
        entity_ids={str(s['entity_id']) for s,_ in quoted if s.get('entity_id')}
        if len(entity_ids)==1:
            primary=[s for s in sources if str(s.get('entity_id','')) in entity_ids and reference_rank(s)>best]
            if primary:
                best=max(reference_rank(s) for s in primary)
                proofs=[s['content'] for s in primary if reference_rank(s)==best]
        supported=set().union(*(scaling_bases(q) for q in proofs))
        for basis,stat in requested:
            alternatives={b for b,r in supported if r==stat}
            if alternatives and basis not in alternatives:
                raise APIError('answer: 加成基数与所引主资料冲突，请区分额外、最大、当前、已损及基础属性')


def validate_condition_strength(claims,sources):
    for claim in claims:
        quoted=_quoted(claim,sources)
        affirmative='；'.join(c for c in re.split(r'[，。；;]',claim['text']) if not _NEGATED.search(c))
        # A quoted sufficient route must not become a necessary/only route.
        for clause in affirmative.split('；'):
            strong=re.search(r'([^，。；]{1,45})是[^，。；]{0,20}(?:前置|必要|必备|唯一)(?:条件|前提)',clause)
            action=_ACTION.search(strong[1]) if strong else None
            if not action:continue
            aliases=r'普攻|普通攻击' if action[0] in {'普攻','普通攻击'} else re.escape(action[0])
            if not any(re.search(aliases,q) and _NECESSARY.search(q) for _,q in quoted):
                raise APIError('answer: 原文的一种触发途径不能改写为必要前置条件或唯一途径')
        shared=bool(re.search(r'(?:都|均|两者|二者|双方).{0,35}(?:解锁|解除.{0,5}(?:封印|限制)|释放前置|施放前提)',affirmative))
        entity_ids={str(s['entity_id']) for s,_ in quoted if s.get('entity_id')}
        if not _UNLOCK.search(affirmative) or not (shared or len(entity_ids)<=1):continue
        for eid in entity_ids or {None}:
            scoped=[(s,q) for s,q in quoted if eid is None or str(s.get('entity_id'))==eid]
            if not scoped:continue
            best=max(reference_rank(s) for s,_ in scoped);proof='\n'.join(q for s,q in scoped if reference_rank(s)==best)
            if re.search(r'刷新.{0,18}冷却|冷却.{0,18}刷新',proof) and not _UNLOCK.search(proof):
                raise APIError('answer: 刷新技能冷却不等于解锁技能或满足释放前提')


def validate_semantic_invariants(claims,sources):
    if any(unsupported_absence_claim(claim) for claim in claims):raise APIError('answer: 资料缺口不能挂无关原文引用，请放入未确认事项')
    validate_scaling_bases(claims,sources)
    validate_condition_strength(claims,sources)
