import hashlib
import json
import re
import shutil
import zipfile
import threading
from pathlib import Path
from .storage import safe_extract

VIDEO_CACHE_LOCK=threading.RLock()

class Library:
    def __init__(self,store,manifest_id,cache):
        self.store=store;self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        store.download(manifest_id,self.cache/'library.json')
        self.manifest=json.loads((self.cache/'library.json').read_text())
        self.videos={str(v['video_id']):v for v in self.manifest['videos']}
        if len(self.videos)!=len(self.manifest['videos']):raise ValueError('Duplicate video IDs in library manifest')

    @classmethod
    def from_folder(cls,store,folder_id,cache):
        obj=object.__new__(cls);obj.store=store;obj.cache=Path(cache);obj.cache.mkdir(parents=True,exist_ok=True)
        items=store.children(folder_id)
        byname={x['name']:x for x in items}
        if ('index.zip' not in byname and 'asset-parts.json' not in byname) or 'sha256sums.json' not in byname:
            raise ValueError('Library needs index.zip and sha256sums.json, produced by prepare_library.py.')
        store.download(byname['sha256sums.json']['id'],obj.cache/'sha256sums.json')
        checks=json.loads((obj.cache/'sha256sums.json').read_text())
        media=list(items)
        for item in items:
            if item['mimeType']=='application/vnd.google-apps.folder' and item['name'].lower() in {'video','videos'}:
                media.extend(store.children(item['id']))
        allbyname={x['name']:x for x in media}
        parts={}
        if 'asset-parts.json' in byname:
            store.download(byname['asset-parts.json']['id'],obj.cache/'asset-parts.json')
            rawparts=json.loads((obj.cache/'asset-parts.json').read_text())
            for name,asset in rawparts.items():
                if not 1<=len(asset.get('parts',[]))<=32:raise ValueError('Invalid number of asset parts')
                if name not in checks or asset['sha256']!=checks[name]:raise ValueError('Asset checksum does not match library checksums')
                resolved=[]
                for part in asset['parts']:
                    if part['name'] not in allbyname:raise ValueError(f'Library upload is incomplete: {name}')
                    if not re.fullmatch('[a-f0-9]{64}',part['sha256']):raise ValueError('Invalid part checksum')
                    resolved.append({'file_id':allbyname[part['name']]['id'],'sha256':part['sha256']})
                parts[name]={'parts':resolved,'sha256':asset['sha256']}
        videos=[]
        for item in media:
            match=re.match(r'^(\d+)(?:_|\.)',item['name'])
            if match and Path(item['name']).suffix.lower() in {'.mp4','.mov','.mkv','.webm','.avi','.m4v'}:
                if item['name'] not in checks:raise ValueError('Source video has no administrator-provided checksum')
                videos.append({'video_id':match.group(1),'filename':item['name'],'file_id':item['id'],'sha256':checks[item['name']]})
        for name,asset in parts.items():
            match=re.match(r'^(\d+)(?:_|\.)',name)
            if match and Path(name).suffix.lower() in {'.mp4','.mov','.mkv','.webm','.avi','.m4v'}:
                videos.append({'video_id':match.group(1),'filename':name,**asset})
        index=parts.get('index.zip') or {'file_id':byname['index.zip']['id'],'sha256':checks['index.zip']}
        obj.manifest={'index':index,'videos':videos}
        if 'article-title-cache.json' in byname:
            if 'article-title-cache.zip' not in byname:raise ValueError('Article title cache upload is incomplete')
            descriptor=obj.cache/'article-title-cache.json'
            store.download(byname['article-title-cache.json']['id'],descriptor)
            info=json.loads(descriptor.read_text())
            obj.manifest['title_cache']={**info,'file_id':byname['article-title-cache.zip']['id']}
        obj.videos={v['video_id']:v for v in videos}
        if len(obj.videos)!=len(videos):raise ValueError('More than one source file uses the same video ID')
        if 'video-overviews.json' in byname:
            if 'video-overviews.zip' not in byname:raise ValueError('Prepared visual overview upload is incomplete')
            descriptor=obj.cache/'video-overviews.json'
            store.download(byname['video-overviews.json']['id'],descriptor)
            obj.manifest['overview_cache']={**json.loads(descriptor.read_text()),'file_id':byname['video-overviews.zip']['id']}
        return obj

    def restore_overviews(self):
        item=self.manifest.get('overview_cache')
        if not item:return
        from .coverage import VERSION,INTERVAL
        expected={str(vid):v['sha256'] for vid,v in self.videos.items()}
        if item.get('version')!=VERSION or item.get('interval')!=INTERVAL or item.get('sources')!=expected:return
        checksum=item.get('sha256','')
        if not re.fullmatch('[a-f0-9]{64}',checksum):raise ValueError('Invalid overview archive checksum')
        folder=self.cache/'full-overviews';folder.mkdir(exist_ok=True)
        marker=folder/'prepared-overviews.sha256'
        if marker.exists() and marker.read_text()==checksum:return
        archive=self.cache/('video-overviews-'+checksum+'.zip')
        if not archive.exists():self.download_asset(item,archive)
        with archive.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=checksum:raise ValueError('Overview archive checksum mismatch')
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist())>500*1024**2:raise ValueError('Overview archive is too large')
            for entry in z.infolist():
                parts=entry.filename.split('/')
                if len(parts)!=3 or parts[0] not in expected or parts[1]!=expected[parts[0]][:16] or not re.fullmatch(r'(?:storyboards\.json|sheet-\d{4}\.jpg)',parts[2]):
                    raise ValueError('Unexpected prepared overview contents')
        safe_extract(archive,folder);marker.write_text(checksum)

    def has_prepared_overview(self,video_id):
        from .coverage import valid_prepared_manifest
        item=self.videos[str(video_id)]
        folder=self.cache/'full-overviews'/str(video_id)/item['sha256'][:16]
        try:
            manifest=json.loads((folder/'storyboards.json').read_text())
            return valid_prepared_manifest(manifest,folder)
        except (OSError,ValueError,KeyError):return False

    def caption_context(self,video_id,start,end):
        import sqlite3
        database=self.index()/'corpus.sqlite'
        connection=sqlite3.connect('file:'+database.as_posix()+'?mode=ro',uri=True)
        try:
            rows=connection.execute('SELECT start_ms,end_ms,text FROM passages WHERE article_id=? AND end_ms>=? AND start_ms<=? ORDER BY start_ms LIMIT 12',
                                    (str(video_id),max(0,(start-5)*1000),(end+5)*1000)).fetchall()
            return [{'start':a/1000,'end':b/1000,'text':text} for a,b,text in rows]
        finally:connection.close()

    def download_asset(self,item,dest):
        dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
        if 'parts' not in item:return self.store.download(item['file_id'],dest)
        with dest.open('wb') as output:
            for part in item['parts']:
                temp=self.cache/'parts'/part['sha256']
                if not temp.exists():self.store.download(part['file_id'],temp)
                with temp.open('rb') as inp:
                    if hashlib.file_digest(inp,'sha256').hexdigest()!=part['sha256']:raise ValueError('Asset-part checksum mismatch')
                    inp.seek(0);shutil.copyfileobj(inp,output,1024*1024)

    def index(self):
        item=self.manifest['index']
        expected=item['sha256']
        if not re.fullmatch('[a-fA-F0-9]{64}',expected):raise ValueError('Invalid index checksum')
        folder=self.cache/expected
        if not (folder/'manifest.json').exists():
            archive=self.cache/'index.zip';self.download_asset(item,archive)
            with archive.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual!=expected:raise ValueError('Index checksum mismatch')
            safe_extract(archive,folder)
        self.restore_title_cache(folder,expected)
        return folder

    def restore_title_cache(self,folder,index_hash):
        item=self.manifest.get('title_cache')
        if not item:return
        expected=item.get('sha256','')
        if not re.fullmatch('[a-f0-9]{64}',expected) or item.get('index_sha256')!=index_hash:
            raise ValueError('Article title cache does not match this caption index')
        marker=folder/'title-cache.sha256'
        if marker.exists() and marker.read_text()==expected:return
        archive=self.cache/('article-titles-'+expected+'.zip')
        if not archive.exists():self.download_asset(item,archive)
        with archive.open('rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=expected:raise ValueError('Article title cache checksum mismatch')
        allowed={'v2_cache/'+name for name in ['article_cache_meta.json','article_ids.json','article_title_embeddings.npy','article_titles.json']}
        with zipfile.ZipFile(archive) as z:
            names=z.namelist()
            if len(names)!=4 or set(names)!=allowed or sum(x.file_size for x in z.infolist())>64*1024**2:
                raise ValueError('Unexpected article title cache contents')
        safe_extract(archive,folder)
        marker.write_text(expected)

    def video(self,video_id):
        with VIDEO_CACHE_LOCK:return self._video(video_id)

    def _video(self,video_id):
        item=self.videos[str(video_id)]
        expected=item.get('sha256')
        if not expected:raise ValueError('Source video must have a recorded checksum')
        safe_id=hashlib.sha256(f'{video_id}:{expected}'.encode()).hexdigest()[:20]
        suffix=Path(item['filename']).suffix.lower()
        if suffix not in {'.mp4','.mov','.mkv','.m4v','.avi','.webm'}:raise ValueError('Unsupported source video')
        path=self.cache/'videos'/(safe_id+suffix)
        if not path.exists():
            temp=path.with_suffix('.download');self.download_asset(item,temp)
            with temp.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual!=expected:raise ValueError('Video checksum mismatch')
            temp.replace(path)
        return path

