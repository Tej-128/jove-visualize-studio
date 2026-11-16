"""Private Google Drive storage; credentials remain in server memory."""
import io
import json
from pathlib import Path
import tempfile
import zipfile
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

BASE='https://www.googleapis.com/drive/v3'

class DriveStore:
    def __init__(self,credential_info):
        creds=service_account.Credentials.from_service_account_info(credential_info,scopes=['https://www.googleapis.com/auth/drive'])
        self.session=AuthorizedSession(creds)

    def request(self,method,url,**kwargs):
        r=self.session.request(method,url,timeout=120,**kwargs);r.raise_for_status();return r

    def info(self,file_id):
        return self.request('GET',f'{BASE}/files/{file_id}',params={'fields':'id,name,mimeType,size,modifiedTime,driveId','supportsAllDrives':'true'}).json()

    def children(self,folder_id):
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+',folder_id):raise ValueError('Invalid Drive folder identifier')
        items=[];token=None
        while True:
            params={'q':f"'{folder_id}' in parents and trashed=false",'fields':'nextPageToken,files(id,name,mimeType,size)',
                    'pageSize':1000,'supportsAllDrives':'true','includeItemsFromAllDrives':'true'}
            if token:params['pageToken']=token
            response=self.request('GET',f'{BASE}/files',params=params).json();items.extend(response.get('files',[]))
            token=response.get('nextPageToken')
            if not token:break
        return items

    def download(self,file_id,dest):
        dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
        meta=self.info(file_id)
        with self.session.get(f'{BASE}/files/{file_id}',params={'alt':'media','supportsAllDrives':'true'},stream=True,timeout=120) as r:
            r.raise_for_status()
            with dest.open('wb') as f:
                for chunk in r.iter_content(1024*1024):f.write(chunk)
        return meta

    def text_document(self,file_id):
        info=self.info(file_id)
        if info['mimeType']=='application/vnd.google-apps.document':
            return self.request('GET',f'{BASE}/files/{file_id}/export',params={'mimeType':'text/plain'}).text
        if info['mimeType']=='application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            from docx import Document
            r=self.request('GET',f'{BASE}/files/{file_id}',params={'alt':'media','supportsAllDrives':'true'})
            d=Document(io.BytesIO(r.content));return '\n'.join(p.text for p in d.paragraphs)
        raise ValueError('The selected file is not a Google Doc or Word document.')

    def upload(self,path,folder_id):
        path=Path(path)
        r=self.request('POST','https://www.googleapis.com/upload/drive/v3/files',
            params={'uploadType':'resumable','supportsAllDrives':'true'},
            json={'name':path.name,'parents':[folder_id]},headers={'X-Upload-Content-Length':str(path.stat().st_size)})
        # Drive issues an HTTPS Google upload URL. Never send credentials to other hosts.
        location=r.headers['Location']
        from urllib.parse import urlsplit
        if urlsplit(location).scheme!='https' or urlsplit(location).hostname not in {'www.googleapis.com','content.googleapis.com'}:
            raise ValueError('Unexpected upload destination')
        with path.open('rb') as f:self.request('PUT',location,data=f,headers={'Content-Length':str(path.stat().st_size)})

    def folder(self,parent_id,name):
        parent=self.info(parent_id)
        if not parent.get('driveId'):
            raise ValueError('Configure OUTPUT_FOLDER_ID as an approved company Shared Drive folder. Service accounts cannot own output files in personal My Drive. No narration or AI generation has started.')
        return self.request('POST',f'{BASE}/files',params={'supportsAllDrives':'true'},
            json={'name':name,'mimeType':'application/vnd.google-apps.folder','parents':[parent_id]}).json()['id']

def safe_extract(archive,dest):
    dest=Path(dest).resolve();dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>3*1024**3:raise ValueError('Index archive is too large')
        for i in z.infolist():
            if not (dest/i.filename).resolve().is_relative_to(dest):raise ValueError('Unsafe archive path')
            if (i.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive links are not allowed')
        z.extractall(dest)

