"""Installed into Strata's serve/ by install-media-guard.py. No GPU dependencies."""
import json
import os
from pathlib import Path
import re
import threading
import uuid
import time

request = threading.local()
DEFAULT_LEASE = 'C:\\Projects\\kurai-llm\\skills\\kurai-3d\\gpu-lease.json'  # Installer records the checkout; KURAI_GPU_LEASE_FILE overrides it after relocation.
DEFAULT_STOP = 'C:\\Projects\\kurai-llm\\Logs\\llm-manual-stop.json'
SECONDARY = 'kurai-film-secondary'
INFERENCE = {'/v1/chat/completions','/v1/messages','/v1/responses','/completion','/completions'}
_readers = {}


def secondary_state(service):
    """Read the running engine's spawn spec, not an editable config's desired state."""
    try:
        exe, args, _, _, env = service.engine.spawn
        env = env or {}
        if any(str(a).startswith(('--peer', '--remote', '--expert-cache-device', '--layer-split', '--helper')) for a in args):
            raise ValueError('Main engine uses a helper/split; secondary rendering requires single GPU v1')
        primary = int(env['STRATA_PRIMARY_DEVICE'])
        vision_env = service.vision.spawn[2]
        secondary = int(vision_env['HIP_VISIBLE_DEVICES'])
        if primary == secondary:
            raise ValueError('Vision and main share a GPU')
        from serve.telemetry import gpu_reader
        for index in (primary,secondary):
            if index not in _readers:
                _readers[index] = gpu_reader(index,amd=True)
        main, aux = _readers[primary], _readers[secondary]
        if main.name() != 'AMD Radeon RX 6800' or aux.name() != 'AMD Radeon RX 6600':
            raise ValueError('This coexistence profile requires verified RX6800 main and RX6600 secondary')
        memory = aux.read()
        free = (memory['mem_total']-memory['mem_used'])/2**20
        resident = service.loaded() and not service._vision_down()
        return {'version':1,'available':bool(resident),'main_pid':service.engine.proc.pid if resident else None,
                'primary_name':main.name(),'secondary_name':aux.name(),'secondary_hip':secondary,
                'secondary_free_mib':round(free),'main_resident':bool(resident)}
    except (KeyError,ValueError,AttributeError,TypeError,ImportError) as error:
        return {'version':1,'available':False,'reason':str(error)}


def cancel_secondary(path,value,reason):
    cancel = path.with_name(path.name+'.cancel-'+value['token']+'.json')
    temporary = cancel.with_name(cancel.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps({'token':value['token'],'reason':reason}),encoding='utf-8')
    os.replace(temporary,cancel)


def require_vision_allowed():
    """A new image gets RX6600 after Blender drains; cached image embeddings need no interruption."""
    found = lease()
    if not found or found[1].get('kind') != SECONDARY:
        return
    path,value = found
    cancel_secondary(path,value,'new_vision_image_requested')
    deadline = time.monotonic()+30
    while lease():
        if time.monotonic()>deadline:
            raise RuntimeError('Secondary render did not drain for vision; retry after its worker stops')
        time.sleep(.1)


def lease():
    filename = os.environ.get('KURAI_GPU_LEASE_FILE') or DEFAULT_LEASE
    if not filename:
        return None
    path = Path(filename)
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict) or not isinstance(value.get('pid'), int) or value['pid'] <= 0 or not value.get('token'):
            raise ValueError('Invalid lease identity')
        return path, value
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise RuntimeError('GPU lease cannot be verified; loading is blocked.') from error


def require_load_allowed(resident=False, secondary=None):
    stop_file = os.environ.get('KURAI_STOP_FILE') or DEFAULT_STOP
    if stop_file and Path(stop_file).exists():
        raise RuntimeError('LLM was manually stopped. Use Start-LLM to resume.')
    found = lease()
    if not found:
        return
    path, value = found
    if value.get('kind') == SECONDARY:
        state = json.loads(Path(value['manifest']).read_text(encoding='utf-8'))
        expected = state.get('secondary',{})
        if (resident and secondary and secondary.get('available') and
                secondary.get('main_pid') == expected.get('main_pid') and
                getattr(request,'path',None) in INFERENCE):
            return
        cancel_secondary(path,value,'main_reload_or_config_change')
        raise RuntimeError('Secondary render is draining before a main reload/config change; retry through proxy')
    token = getattr(request, 'token', os.environ.get('KURAI_SHIFT_TOKEN'))
    if token and token == value.get('token'):
        if value.get('kind') == 'kurai-llm-maintenance':
            return
        state = json.loads(Path(value['manifest']).read_text(encoding='utf-8'))
        if state.get('job') == token and state.get('phase') == 'restoring' and state.get('children_stopped') is True:
            return
    if value.get('kind') in ('kurai-image', 'kurai-3d') and re.fullmatch('[0-9a-f]{32}', str(value.get('token', ''))):
        cancel = path.with_name(path.name + '.cancel-' + value['token'] + '.json')
        temporary = cancel.with_name(cancel.name + '.' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps({'token': value['token'], 'reason': 'direct_backend_requested'}), encoding='utf-8')
        os.replace(temporary, cancel)
    raise RuntimeError('GPU belongs to a media shift; cancellation/recovery started. Retry through Kurai proxy.')


def guard_request(path, headers, query=''):
    request.token = headers.get('X-Kurai-Shift-Token')
    request.path = path
    # Status/tokenizer reads use GET. Only memory-releasing controls and snapshot SAVE are allowed here.
    if path in ('/unload', '/v1/unload') or path.startswith('/slots/') and query == 'action=save':
        return
    found = lease()
    if found and found[1].get('kind') == SECONDARY and path in INFERENCE:
        return  # ensure_loaded verifies the actual resident engine before generation.
    require_load_allowed()
