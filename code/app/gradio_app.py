"""Run with app/.venv/bin/python app/gradio_app.py (from any directory)."""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Set before importing Agent -> NumPy. No local model is loaded by this app.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from agent.config import Settings
from agent.conversation import history_from_messages
from agent.official import load_store
from app.backend import DemoBackend, EXAMPLES, available_models, markdown_text, source_file
from app.session import save_session, restore_session
from app.sharing import ShareManager,local_control

CSS = (Path(__file__).with_name("theme.css")).read_text(encoding="utf-8")

# Code-native artwork: no external font, image request, or extra model call.
MARK = '''<svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
<path d="M24 5 43 24 24 43 5 24 24 5Z" stroke="currentColor" stroke-width="1.2"/>
<path d="m24 12 5 12-5 12-5-12 5-12Z" fill="currentColor"/>
<path d="m12 24 12-5 12 5-12 5-12-5Z" fill="currentColor"/>
<circle cx="24" cy="24" r="18" stroke="currentColor" stroke-opacity=".25"/>
</svg>'''
WELCOME = f'''<div class="chat-welcome">
<div class="welcome-mark">{MARK}</div>
<div class="welcome-eyebrow">你的峡谷知识搭子</div>
<h2>少一点困惑，<span>多一点胜算。</span></h2>
<p>从英雄技能到活动规则，陪你把问题聊明白。</p>
<div class="welcome-tags"><span>知识库优先</span><b>·</b><span>必要时联网</span><b>·</b><span>回答附来源</span></div>
</div>'''

EXAMPLE_LABELS = [
    "元宵活动漏签，能补签吗？", "端午活动怎么组队参与？",
    "铠的二技能有什么控制？", "总结妲己的各个技能",
    "看看本周有哪些新活动", "账号被封了，该怎么办？",
]


def chat_turn(backend, question, model, threshold, conversation, messages):
    """Only presentation streams; the original evidence-checked Agent runs once."""
    question = question.strip() if isinstance(question, str) else ""
    if not 1 <= len(question) <= 2000:
        gr.Warning("请输入 1–2000 字的问题。")
        yield messages or [], conversation, gr.update(), gr.update(), gr.update()
        return
    transcript = list(messages or [])
    transcript.append({"role": "user", "content": markdown_text(question)})
    pending = {"role": "assistant", "content": "正在检索资料、核验来源，请稍候。",
               "metadata": {"title": "正在查证资料…", "status": "pending"}}
    yield transcript + [pending], conversation, gr.update(value="", interactive=False), gr.update(interactive=False), gr.update(interactive=False)
    history = history_from_messages(messages, conversation)
    answer, route, sources, metrics, state = backend.ask_session(question, model, threshold, history)
    transcript.extend([
        {"role": "assistant", "content": answer + "\n\n---\n\n**参考来源**\n\n" + sources},
        {"role": "assistant", "content": route,
         "metadata": {"title": "路由决策", "status": "done"}},
        {"role": "assistant", "content": metrics,
         "metadata": {"title": "耗时与用量 · 费用估算", "status": "done"}},
    ])
    yield transcript, state, gr.update(value="", interactive=True), gr.update(interactive=True), gr.update(interactive=True)


def build_demo(backend: DemoBackend, settings: Settings, sharing: ShareManager | None = None) -> gr.Blocks:
    models=available_models(settings)
    default_model = settings.model if settings.model in models else next(iter(models))
    with gr.Blocks(title="王者荣耀客服 · Agentic RAG", analytics_enabled=False) as demo:
        conversation=gr.State([])
        saved_conversation=gr.BrowserState({},storage_key='king-rag-conversation-v6')
        with gr.Sidebar(width=248, elem_id="settings-sidebar"):
            gr.HTML(f'<div class="sidebar-brand"><span class="brand-mark">{MARK}</span><div>王者 · 问答<small>GAME KNOWLEDGE ASSISTANT</small></div></div>')
            clear = gr.Button("＋  开启新对话", variant="secondary", elem_id="new-chat")
            gr.Markdown("本机最多保留20轮，24小时后失效。\n开启新对话可清除。", elem_classes="quiet-note")
            gr.HTML('<div class="sidebar-section-label">偏好设置 <span>PREFERENCES</span></div>')
            model = gr.Dropdown(choices=[(label, value) for value, label in models.items()],
                                value=default_model, label="回答模型", allow_custom_value=False,
                                elem_id="model-picker")
            gr.Markdown("模型切换对下一次提问生效。", elem_classes="quiet-note")
            with gr.Accordion("检索设置", open=False, elem_id="retrieval-settings"):
                threshold = gr.Slider(0, 1, value=settings.kb_threshold, step=0.05,
                                      label="知识库阈值")
                gr.Markdown("证据不足时联网兜底。分数不代表正确率，设置对下一次提问生效。", elem_classes="quiet-note")
            store=load_store(settings.root)
            official=(f"\n\n官网目录 {len(store.heroes)} 个英雄条目 · 技能资料 {len(store.by_entity)} 个英雄条目"
                      f"\n\n装备 {len(store.items)} 项 · 召唤师技能 {len(store.payload.get('summoners',[]))} 项") if store else ""
            gr.HTML('<div class="knowledge-card"><div class="knowledge-heading"><span class="small-diamond">◇</span> 你的游戏资料库</div><div class="knowledge-count">110 <span>篇活动与基础知识</span></div><div class="knowledge-caption">活动规则 / 英雄技能 / 玩法机制</div></div>')
            with gr.Accordion("查看资料概览", open=False, elem_id="knowledge-overview"):
                gr.Markdown("50 篇活动规则 · 60 篇基础知识（演示资料）"+official)
            if sharing is not None:
                with gr.Accordion("临时分享 · 给朋友试用", open=False) as share_panel:
                    gr.Markdown("开启后生成链接和访问密码，2小时后自动关闭。\n分享期间请保持 Mac 开机、联网且不休眠。", elem_classes="quiet-note")
                    share_start=gr.Button("开启临时分享",variant="secondary")
                    share_stop=gr.Button("关闭临时分享",variant="secondary")
                    share_details=gr.Textbox(value="尚未开启临时分享。",label="发给朋友的访问说明",lines=6,interactive=False,buttons=['copy'])
                def start_share(request: gr.Request):
                    if not local_control(request):raise gr.Error('临时分享只能在分享者的本机页面操作。')
                    return sharing.start()
                def stop_share(request: gr.Request):
                    if not local_control(request):raise gr.Error('临时分享只能在分享者的本机页面操作。')
                    return sharing.stop()
                def share_view(request: gr.Request):
                    local=local_control(request)
                    return gr.update(visible=local),sharing.describe() if local else ''
                share_start.click(start_share,outputs=share_details,queue=False,api_name=False,show_progress='minimal')
                share_stop.click(stop_share,outputs=share_details,queue=False,api_name=False,show_progress='minimal')
                demo.load(share_view,outputs=[share_panel,share_details],queue=False,api_name=False)
            gr.HTML('<div class="sidebar-foot"><div><span class="privacy-dot"></span> 本机界面 · API 模型服务</div><p>历史活动按收录版本回答<br>实时信息请以官方公告为准</p><small>非官方演示助手</small></div>')

        with gr.Column(elem_id="chat-workspace"):
            gr.HTML('<header class="page-header"><div class="header-title"><span class="header-symbol">◇</span><h1>峡谷会客厅</h1><span class="header-divider">/</span><span class="header-subtitle">王者荣耀客服助手</span></div><span class="header-badge"><i></i> 回答有据可查</span></header>')
            chatbot = gr.Chatbot(value=[], label="对话", show_label=False, height=400,
                                 min_height=200, layout="bubble", placeholder=WELCOME,
                                 buttons=["copy"], feedback_options=(), sanitize_html=True,
                                 allow_tags=False, allow_file_downloads=False, elem_id="chat-panel")
            question = gr.Textbox(label="用户提问", show_label=False, container=False,
                                  placeholder="想了解什么？也可以接着上一条追问…", lines=1,
                                  max_lines=5, max_length=2000, scale=9, min_width=180,
                                  elem_id="question-input", render=False)
            with gr.Column(elem_id="quick-start", min_width=0):
                gr.HTML('<div class="examples-heading"><span>从一个问题开始</span><small>点击填入，按发送开始对话</small></div>')
                gr.Examples(examples=[[q] for q in EXAMPLES], inputs=question,
                            label="", example_labels=EXAMPLE_LABELS, elem_id="question-examples",
                            examples_per_page=6, cache_examples=False, run_on_click=False)
            with gr.Column(elem_id="composer", min_width=0):
                with gr.Row(equal_height=True):
                    question.render()
                    submit = gr.Button("发送 ↑", variant="primary", scale=0, min_width=84, elem_id="send-message")
                gr.HTML('<div class="composer-caption"><span>◇ &nbsp;知识库优先 · 必要时联网</span><small><kbd>Enter</kbd> 发送 <span class="key-separator">/</span> <kbd>Shift + Enter</kbd> 换行</small></div>')
            gr.Markdown("每个答案，都值得有个出处。重要信息请以官方公告为准。", elem_id="composer-hint")
        inputs, outputs = [question, model, threshold, conversation, chatbot], [chatbot, conversation, question, submit, clear,saved_conversation]
        def respond(question, model, threshold, conversation, messages):
            for output in chat_turn(backend, question, model, threshold, conversation, messages):
                saved=save_session(output[0],output[1]) if output[2].get('interactive') is True else gr.update()
                yield (*output,saved)
        options = dict(fn=respond, inputs=inputs, outputs=outputs,
                       concurrency_id="agentic-rag", concurrency_limit=1,
                       trigger_mode="once", api_visibility="undocumented", show_progress="minimal")
        submit.click(**options, api_name="ask")
        question.submit(**options, api_name="ask_enter")
        clear.click(fn=lambda: ([], [], "",{}), outputs=[chatbot, conversation, question,saved_conversation],
                    concurrency_id="agentic-rag", concurrency_limit=1, api_name=False)
        chatbot.clear(fn=lambda: ([], [], "",{}), outputs=[chatbot, conversation, question,saved_conversation],
                      concurrency_id="agentic-rag", concurrency_limit=1, api_name=False)
        # Open navigation on desktop, dismiss the drawer on an initial phone
        # visit. This is a local UI action and never calls the Agent or an API.
        demo.load(fn=restore_session, inputs=saved_conversation, outputs=[chatbot,conversation,saved_conversation], queue=False, api_name=False,
                  js="""(saved) => {
                    const sidebar = document.getElementById('settings-sidebar');
                    if (window.matchMedia('(max-width: 760px)').matches && sidebar?.classList.contains('open')) {
                      sidebar.querySelector('button[aria-label="Toggle Sidebar"]')?.click();
                    }
                    return [saved];
                  }""")
    return demo.queue(default_concurrency_limit=1, max_size=4,api_open=False)


def create_app(env_file: Path | None = None, root: Path = ROOT, sharing: ShareManager | None = None) -> FastAPI:
    settings = Settings.load(env_file, root=root)
    demo = build_demo(DemoBackend(root=root, env_file=env_file), settings,sharing)
    @asynccontextmanager
    async def lifespan(server):
        try:yield
        finally:
            if sharing is not None:sharing.close()
    server = FastAPI(title="Agentic RAG Demo", docs_url=None, redoc_url=None, openapi_url=None,lifespan=lifespan)

    @server.get("/sources/{doc_id}", response_class=PlainTextResponse)
    def get_source(doc_id: str):
        path = source_file(root, doc_id)
        if path is None:
            raise HTTPException(status_code=404, detail="来源不存在")
        return PlainTextResponse(path.read_text(encoding="utf-8"),
                                 headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})

    # Never expose the project directory through Gradio's arbitrary file route.
    # The narrow /sources/{doc_id} route above serves only KB/policy text.
    return gr.mount_gradio_app(
        server, demo, path="/", server_name="127.0.0.1",
        blocked_paths=[str(root.resolve())], allowed_paths=[],
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate",
                             font=["-apple-system", "BlinkMacSystemFont", "PingFang SC", "sans-serif"]),
        css=CSS, footer_links=[], run_history=False,
    )


def main():
    parser = argparse.ArgumentParser(description="本机 Agentic RAG Gradio 演示")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--env-file", type=Path, help="额外配置文件，优先级仍低于 shell 环境变量")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1–65535 之间")
    try:
        server = create_app(args.env_file,sharing=ShareManager(args.port))
    except (ValueError, OSError):
        parser.exit(1, "配置或资料文件无法读取，请检查 .env 和现有 Agent 数据。\n")
    print(f"\n演示地址：http://127.0.0.1:{args.port}\n停止：Ctrl+C（不会停止已有 Embedding 服务）\n", flush=True)
    uvicorn.run(server, host="127.0.0.1", port=args.port, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
