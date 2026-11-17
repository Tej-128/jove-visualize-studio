"""Authenticate owner and script identity before restoring a cloud checkpoint."""
import hashlib
import hmac
import json

def signature(state,config):
    secret=config.get('google_service_account',{}).get('private_key')
    if not secret:return None
    key=hmac.new(secret.encode(),b'JoVEStudioCheckpoint/v1',hashlib.sha256).digest()
    payload=json.dumps([state.get('id'),state.get('owner'),state.get('input_sha256')],separators=(',',':')).encode()
    return hmac.new(key,payload,hashlib.sha256).hexdigest()

def valid(state,config):
    expected=signature(state,config);received=state.get('checkpoint_signature')
    return bool(expected and isinstance(received,str) and hmac.compare_digest(expected,received))
