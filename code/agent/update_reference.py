"""Low-priority official refresh: validate candidate first, atomically publish last.

No chat, Search or Fetch API calls. Only official HTTPS and the existing local
embedding server. Failed candidates never replace the active release.
"""
from __future__ import annotations
import argparse, datetime, fcntl, json, os, tempfile, time,plistlib,subprocess,sys,shutil
from pathlib import Path
from urllib.parse import urlsplit

from .config import ROOT,Settings
from .official import digest,load_store,now,_cached_store

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix='.'+path.name,delete=False) as file:
        temp=Path(file.name)
        json.dump(value,file,ensure_ascii=False,indent=2);file.write('\n');file.flush();os.fsync(file.fileno())
    try:os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)

def validate_candidate(root,pointer,previous):
    import numpy as np
    candidate=_cached_store(str(root),json.dumps(pointer,sort_keys=True))
    folder=root/'data/official/releases'/candidate.version
    chunks=json.loads((folder/'chunks.json').read_text());matrix=np.load(folder/'vectors.npy',allow_pickle=False,mmap_mode='r')
    if matrix.ndim!=2 or matrix.shape[0]!=len(chunks) or matrix.shape[1]!=candidate.payload['embedding_dim'] or not np.isfinite(matrix).all():raise ValueError('候选向量不完整')
    if np.any(np.abs(np.linalg.norm(matrix,axis=1)-1)>.01):raise ValueError('候选向量未有效归一化')
    docs={d['doc_id'] for d in candidate.docs}
    if len(docs)!=len(candidate.docs) or len({c['chunk_id'] for c in chunks})!=len(chunks) or any(c['doc_id'] not in docs for c in chunks):raise ValueError('候选文档或切片关联非法')
    if {str(h['ename']) for h in candidate.heroes}!=set(candidate.by_entity):raise ValueError('候选英雄资料未完整覆盖目录')
    if any(len(d.get('skills',[]))<4 for d in candidate.by_entity.values()):raise ValueError('候选技能正文缺失')
    if previous:
        if candidate.payload['embedding_model']!=previous.payload['embedding_model'] or matrix.shape[1]!=previous.payload['embedding_dim']:raise ValueError('自动更新不能切换向量模型')
        # An unexpected deletion needs review; an HTTP200/error body is not a new catalog.
        if not {str(h['ename']) for h in previous.heroes}<={str(h['ename']) for h in candidate.heroes}:raise ValueError('官方目录出现删除，保留旧版待核对')
        stable={d['doc_id'] for d in previous.docs if d.get('collection')!='system_news'}
        if not stable<=docs:raise ValueError('自动更新不能静默删除已有基础资料')
    return candidate

def prune_managed_releases(root,keep=3):
    """Bound only releases created by this updater; never delete old user data."""
    folder=Path(root)/'data/official';releases=folder/'releases';protected=set()
    for name in ['current.json','previous.json']:
        if (folder/name).exists():protected.add(json.loads((folder/name).read_text())['version'])
    managed=[]
    for release in releases.iterdir():
        marker=release/'.refresh-owned.json'
        if not release.is_dir() or release.is_symlink() or not marker.is_file():continue
        try:owned=json.loads(marker.read_text()).get('managed_by')=='agent.update_reference'
        except (OSError,ValueError):continue
        if not owned:continue
        if (release/'.pinned-evaluation').exists():protected.add(release.name)
        managed.append(release)
    managed.sort(key=lambda p:(p/'.refresh-owned.json').stat().st_mtime,reverse=True)
    protected.update(p.name for p in managed[:keep]);removed=[]
    for release in managed:
        if release.name in protected:continue
        if release.resolve().parent!=releases.resolve():continue
        shutil.rmtree(release);removed.append(release.name)
    return removed

def refresh(root=ROOT,*,prepared=None):
    root=Path(root);folder=root/'data/official';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'update.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'skipped','reason':'another_refresh_running'}
        settings=Settings.load(root=root)
        if urlsplit(settings.embedding_url).hostname not in {'localhost','127.0.0.1','::1'}:raise ValueError('自动更新仅允许现有本地向量服务')
        try:os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
        except (AttributeError,OSError):pass
        status_file=folder/'update_status.json'
        old_status=json.loads(status_file.read_text()) if status_file.exists() else {}
        pointer_file=folder/'current.json';old_pointer=json.loads(pointer_file.read_text()) if pointer_file.exists() else None
        pointer_hash=digest(pointer_file.read_bytes()) if old_pointer else None
        state={'status':'running','last_attempt':now(),'last_success':old_status.get('last_success'),
          'active_version':old_pointer['version'] if old_pointer else None,'chat_model_calls':0,'search_calls':0,'fetch_api_calls':0}
        atomic_json(status_file,state)
        start=time.monotonic()
        try:
            previous=load_store(root)
            from .ingest_reference import prepare,build
            if prepared:
                candidate_pointer=build(root,Path(prepared),settings,publish=False)
                gaps_file=Path(prepared)/'basic_refresh_gaps.json'
                gaps=json.loads(gaps_file.read_text()) if gaps_file.exists() else []
            else:
                with tempfile.TemporaryDirectory(prefix='.refresh-',dir=folder) as work:
                    prepare(Path(work),settings)
                    gaps=json.loads((Path(work)/'basic_refresh_gaps.json').read_text())
                    news_audit=json.loads((Path(work)/'system_news_audit.json').read_text())
                    candidate_pointer=build(root,Path(work),settings,publish=False)
            candidate=validate_candidate(root,candidate_pointer,previous)
            from .release_checks import check_reference_release
            release_check=check_reference_release(candidate,settings)
            atomic_json(folder/'releases'/candidate.version/'release_checks.json',release_check)
            if release_check['failed']:raise ValueError('候选资料未通过离线问答门禁，保留旧版')
            if (digest(pointer_file.read_bytes()) if pointer_file.exists() else None)!=pointer_hash:raise ValueError('期间资料版本已被其他操作切换，取消发布')
            if old_pointer:atomic_json(folder/'previous.json',old_pointer)
            atomic_json(folder/'releases'/candidate.version/'.refresh-owned.json',{'managed_by':'agent.update_reference','created_at':now()})
            # This is the only step that changes what new requests see.
            atomic_json(pointer_file,candidate_pointer)
            state.update(status='success',last_success=now(),active_version=candidate.version,
              previous_version=old_pointer['version'] if old_pointer else None,guide_refresh_gaps=gaps,
              hero_entries=len(candidate.heroes),documents=len(candidate.docs),
              reused_vectors=candidate.payload['coverage'].get('reused_vectors',0),
              embedded_vectors=candidate.payload['coverage'].get('embedded_vectors',0))
            if not prepared:state['system_news']=news_audit
            state['release_checks']=release_check
            try:state['pruned_managed_releases']=prune_managed_releases(root)
            except OSError:state['maintenance_warning']='旧索引清理未完成；当前已验证版本不受影响'
        except Exception as error:
            state.update(status='failed',error_type=type(error).__name__,reason='候选更新未通过，继续使用原资料；详情请本地检查来源及服务。')
            # Keys/remote error bodies never enter this status file.
        state['elapsed_seconds']=round(time.monotonic()-start,2);atomic_json(status_file,state)
        return state

def schedule_plist(root,python):
    root=Path(root);logs=root/'data/official'
    return {'Label':'com.example.game-rag.refresh',
      'ProgramArguments':[str(python),'-B','-m','agent.update_reference','--root',str(root)],
      'WorkingDirectory':str(root),'StartCalendarInterval':{'Hour':5,'Minute':30},'RunAtLoad':False,
      'ProcessType':'Background','Nice':10,'LowPriorityIO':True,'ThrottleInterval':3600,
      'EnvironmentVariables':{'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'},
      'StandardOutPath':str(logs/'scheduled_update.log'),'StandardErrorPath':str(logs/'scheduled_update_error.log')}

def install_schedule(root,python):
    root=Path(root);(root/'data/official').mkdir(parents=True,exist_ok=True)
    path=Path.home()/'Library/LaunchAgents/com.example.game-rag.refresh.plist'
    path.parent.mkdir(parents=True,exist_ok=True);data=plistlib.dumps(schedule_plist(root,python))
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as file:file.write(data);temp=Path(file.name)
    os.chmod(temp,0o600);os.replace(temp,path)
    service=f'gui/{os.getuid()}/com.example.game-rag.refresh'
    if subprocess.run(['launchctl','print',service],capture_output=True).returncode==0:
        subprocess.run(['launchctl','bootout',service],check=True,capture_output=True)
    subprocess.run(['launchctl','bootstrap',f'gui/{os.getuid()}',str(path)],check=True,capture_output=True)
    subprocess.run(['launchctl','print',service],check=True,capture_output=True)
    return {'status':'scheduled','daily_local_time':'05:30','plist_path':str(path),'run_at_load':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--prepared-dir',type=Path);parser.add_argument('--install-schedule',action='store_true');args=parser.parse_args()
    result=install_schedule(args.root,sys.executable) if args.install_schedule else refresh(args.root,prepared=args.prepared_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
