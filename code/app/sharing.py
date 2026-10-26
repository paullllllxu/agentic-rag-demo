"""Opt-in, password-protected temporary access to the existing local queue.

Only a small loopback HTTP/SSE gateway is added. It forwards to the original
Gradio process, so public and local questions share the same concurrency limit.
No keys, credentials, tunnel tokens or conversation data are written to disk.
"""
from __future__ import annotations
import atexit,base64,binascii,secrets,socket,tempfile,threading,time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass,field
from datetime import datetime,timedelta
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
import uvicorn
from fastapi import FastAPI,Request
from fastapi.responses import PlainTextResponse,StreamingResponse
from starlette.background import BackgroundTask

PUBLIC_MARKER='x-rag-temporary-share'
HOP_HEADERS={'connection','keep-alive','proxy-authenticate','proxy-authorization','te','trailer','transfer-encoding','upgrade'}
MAX_BODY=2*1024*1024

def local_control(request):
    """The gateway always overwrites the marker; public clients cannot omit it."""
    if request is None or request.headers.get(PUBLIC_MARKER):return False
    if getattr(request.client,'host',None) not in {'127.0.0.1','::1','localhost','testclient'}:return False
    try:
        host=urlsplit('//'+request.headers.get('host','')).hostname
        origin=request.headers.get('origin')
        return host in {'127.0.0.1','localhost','::1','testserver'} and (not origin or urlsplit(origin).netloc==request.headers.get('host'))
    except ValueError:return False

def gateway_app(port:int,username:str,password:str,*,transport=None):
    if not 1<=port<=65535 or not username or len(password)<16:raise ValueError('Invalid share gateway configuration')
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(transport=transport,follow_redirects=False,trust_env=False,
            timeout=httpx.Timeout(10,read=600),limits=httpx.Limits(max_connections=16,max_keepalive_connections=4)) as client:
            app.state.client=client;yield
    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    failures=deque()

    @app.middleware('http')
    async def protect(request,call_next):
        valid=False
        header=request.headers.get('authorization','')
        if len(header)<2048 and header.lower().startswith('basic '):
            try:
                raw=base64.b64decode(header[6:],validate=True).decode('utf-8')
                user,secret=raw.split(':',1)
                valid=secrets.compare_digest(user.encode(),username.encode()) & secrets.compare_digest(secret.encode(),password.encode())
            except (ValueError,UnicodeError,binascii.Error):pass
        if not valid:
            current=time.monotonic()
            while failures and current-failures[0]>60:failures.popleft()
            if len(failures)>=30:return PlainTextResponse('尝试次数较多，请一分钟后重试。',429,headers={'Retry-After':'60','Cache-Control':'no-store'})
            failures.append(current)
            return PlainTextResponse('请输入分享账号和访问密码。',401,
                headers={'WWW-Authenticate':'Basic realm="Temporary game assistant", charset="UTF-8"','Cache-Control':'no-store'})
        if request.method not in {'GET','HEAD','OPTIONS'}:
            origin=request.headers.get('origin')
            if origin and urlsplit(origin).netloc!=request.headers.get('host'):
                return PlainTextResponse('不接受来自其他网站的操作。',403)
        return await call_next(request)

    @app.api_route('/{path:path}',methods=['GET','HEAD','POST','PUT','PATCH','DELETE','OPTIONS'])
    async def forward(request:Request,path:str):
        try:
            declared=int(request.headers.get('content-length','0'))
        except ValueError:return PlainTextResponse('请求格式不正确。',400)
        if declared<0 or declared>MAX_BODY:return PlainTextResponse('本次内容过长。',413)
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>MAX_BODY:return PlainTextResponse('本次内容过长。',413)
        headers={k:v for k,v in request.headers.items() if k.lower() not in HOP_HEADERS|{
            'authorization','content-length','accept-encoding','forwarded','x-forwarded-for','x-forwarded-proto','x-forwarded-host',PUBLIC_MARKER}}
        headers.update({PUBLIC_MARKER:'1','accept-encoding':'identity','x-forwarded-proto':'https',
                        'x-forwarded-host':request.headers.get('host',''),'host':f'127.0.0.1:{port}'})
        query=request.scope.get('query_string',b'')
        raw_path=request.scope.get('raw_path',b'/')+(b'?'+query if query else b'')
        target=httpx.URL(scheme='http',host='127.0.0.1',port=port,raw_path=raw_path)
        client=request.app.state.client
        try:
            upstream=await client.send(client.build_request(request.method,target,headers=headers,content=bytes(body)),stream=True)
        except httpx.HTTPError:
            return PlainTextResponse('本机助手暂时未连接，请联系分享者。',502)
        response=StreamingResponse(upstream.aiter_raw(),status_code=upstream.status_code,
                                   background=BackgroundTask(upstream.aclose))
        response.raw_headers=[(k,v) for k,v in upstream.headers.raw if k.decode().lower() not in HOP_HEADERS|{'cache-control','x-accel-buffering'}]
        response.raw_headers.extend([(b'cache-control',b'no-store'),(b'x-accel-buffering',b'no')])
        return response
    return app

@dataclass(repr=False)
class ShareSession:
    username:str='friend'
    password:str=field(default_factory=lambda:secrets.token_urlsafe(18))
    token:str=field(default_factory=lambda:secrets.token_urlsafe(32))
    cancelled:threading.Event=field(default_factory=threading.Event)
    ready:threading.Event=field(default_factory=threading.Event)
    server:object=None
    server_thread:object=None
    tunnel:object=None
    timer:object=None
    directory:object=None
    url:str=''
    error:bool=False
    expires_at:object=None

def official_tunnel(session,port):
    # Isolate the private adapter; version is pinned by app/requirements.txt.
    # Download/checksum/TLS behavior is the same as Gradio's own share feature.
    from gradio import networking
    from gradio.tunneling import Tunnel
    response=httpx.get(networking.GRADIO_API_SERVER,timeout=15)
    response.raise_for_status();config=response.json()[0]
    session.directory=tempfile.TemporaryDirectory(prefix='rag-share-')
    certificate=Path(session.directory.name)/'root.pem'
    certificate.write_text(config['root_ca']);certificate.chmod(0o600)
    return Tunnel(config['host'],int(config['port']),'127.0.0.1',port,session.token,str(certificate))

class ShareManager:
    def __init__(self,port=7860,*,lifetime=7200,start_timeout=55,tunnel_factory=official_tunnel):
        self.port=port;self.lifetime=lifetime;self.start_timeout=start_timeout;self.tunnel_factory=tunnel_factory
        self.lock=threading.RLock();self.session=None;self.last_message='尚未开启临时分享。'
        atexit.register(self.close)

    def _cleanup(self,session):
        session.cancelled.set()
        if session.timer:session.timer.cancel()
        tunnel=session.tunnel
        if tunnel:
            process=getattr(tunnel,'proc',None)
            try:tunnel.kill()
            except (OSError,RuntimeError):pass
            if process:
                try:process.wait(timeout=2)
                except Exception:
                    try:process.kill()
                    except OSError:pass
        if session.server:session.server.should_exit=True
        thread=session.server_thread
        if thread and thread is not threading.current_thread():thread.join(timeout=2)
        if session.directory:
            try:session.directory.cleanup()
            except OSError:pass

    def _run(self,session):
        try:
            sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.listen(16)
            server=uvicorn.Server(uvicorn.Config(gateway_app(self.port,session.username,session.password),
                host='127.0.0.1',port=port,log_level='critical',access_log=False,limit_concurrency=24,timeout_keep_alive=5))
            session.server=server
            session.server_thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True,name='temporary-share-gateway')
            session.server_thread.start()
            deadline=time.monotonic()+5
            while not server.started and time.monotonic()<deadline and not session.cancelled.wait(.05):pass
            if not server.started or session.cancelled.is_set():raise RuntimeError('Gateway not ready')
            session.tunnel=self.tunnel_factory(session,port)
            if session.cancelled.is_set():raise RuntimeError('Share cancelled')
            url=session.tunnel.start_tunnel().strip();parsed=urlsplit(url)
            if parsed.scheme!='https' or not (parsed.hostname or '').endswith('.gradio.live') or parsed.username or parsed.password:
                raise ValueError('Untrusted share URL')
            with self.lock:
                if session.cancelled.is_set() or self.session is not session:raise RuntimeError('Share cancelled')
                session.url=url;session.expires_at=datetime.now(ZoneInfo('Asia/Shanghai'))+timedelta(seconds=self.lifetime)
                session.timer=threading.Timer(self.lifetime,lambda:self.stop(expected=session,expired=True));session.timer.daemon=True;session.timer.start()
        except Exception:
            session.error=True
            with self.lock:
                if self.session is session:self.last_message='临时分享未连接成功，请稍后重试；本机助手仍可使用。'
            self._cleanup(session)
        finally:
            if session.cancelled.is_set():self._cleanup(session)
            session.ready.set()

    def start(self):
        with self.lock:
            if self.session and not self.session.cancelled.is_set():session=self.session
            else:
                session=ShareSession();self.session=session
                threading.Thread(target=self._run,args=(session,),daemon=True,name='temporary-share-start').start()
        if not session.ready.wait(self.start_timeout):
            self.stop(expected=session)
            with self.lock:self.last_message='临时分享连接超时，已关闭这次尝试；本机助手仍可使用。'
        return self.describe()

    def describe(self):
        with self.lock:
            session=self.session
            if not session or session.cancelled.is_set() or session.error:return self.last_message
            if not session.url:return '正在连接临时分享服务，请稍候。'
            return (f'访问地址：{session.url}\n账号：{session.username}\n访问密码：{session.password}\n'
                    f'自动关闭：{session.expires_at:%m月%d日 %H:%M}（北京时间）\n'
                    '用浏览器打开地址，按提示输入账号和密码。请勿转发给其他人。')

    def stop(self,*,expected=None,expired=False):
        with self.lock:
            session=self.session
            if expected is not None and session is not expected:return self.last_message
            self.session=None;self.last_message='临时分享已到期并关闭。' if expired else '临时分享已关闭，原地址和密码已失效。'
        if session:self._cleanup(session)
        return self.last_message

    def close(self):self.stop()
