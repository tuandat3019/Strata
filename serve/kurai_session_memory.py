"""Retry a refused native SAVE after reclaiming only this service's memory.

Installed as serve/kurai_session_memory.py. The caller holds Service.fifo.
Never discard the live KV, lower the engine's memory floor, or touch other apps.
"""
import ctypes
from ctypes import wintypes
import gc
import os
import re
import time

SETTLE_SECONDS = 30
MAX_RETRIES = 3


def required_mib(error):
    match = re.search(r'\((\d+) MiB plus a floor of (\d+) MiB needed,', str(error))
    # Diagnostics round down; reserve another 2 MiB rather than weakening admission.
    return sum(map(int, match.groups())) + 2 if match else None


def retryable(action, error):
    return (action == 'save' and getattr(error, 'kind', None) == 'memory'
            and not getattr(error, 'published', False)
            and str(error).startswith('not enough RAM to save the session'))


def wait_for_headroom(service, engine, process, error, deadline):
    needed = required_mib(error)
    while service.engine is engine and engine.proc is process and process.poll() is None:
        available = available_mib()
        if needed is None or available is None or available >= needed:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.5, remaining))
    return False


def available_mib():
    if os.name != 'nt':
        return None
    class Memory(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ('total_phys', 'avail_phys', 'total_page',
                'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]
    state = Memory()
    state.length = ctypes.sizeof(state)
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(Memory)]
    api.GlobalMemoryStatusEx.restype = wintypes.BOOL
    return state.avail_phys // 2**20 if api.GlobalMemoryStatusEx(ctypes.byref(state)) else None


def trim_working_set(process=None):
    """Use the already-owned Popen HANDLE; never reopen a potentially reused PID.

    Windows can page out pageable memory/file mappings while all virtual allocations
    and the live session remain valid. Locked/pinned allocations may stay resident.
    """
    if os.name != 'nt':
        return False
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.GetCurrentProcess() if process is None else getattr(process, '_handle', None)
    if handle is None or process is not None and process.poll() is not None:
        return False
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
    psapi.EmptyWorkingSet.restype = wintypes.BOOL
    return bool(psapi.EmptyWorkingSet(handle))


def reclaim(service):
    before = available_mib()
    vision = getattr(service, 'vision', None)
    released = False
    if vision is not None and hasattr(vision, 'unload') and hasattr(vision, 'alive') and vision.alive():
        # Encoded images are already cached on disk; unloading the encoder keeps them.
        vision.unload()
        released = True
    gc.collect()
    engine_trimmed = trim_working_set(getattr(service.engine, 'proc', None))
    server_trimmed = trim_working_set()
    print('[kurai-session-memory] SAVE memory recovery: '
          f'available={before}->{available_mib()} MiB; vision_released={released}; '
          f'engine_trimmed={engine_trimmed}; server_trimmed={server_trimmed}', flush=True)


def session_file(service, action, path, refused_type):
    try:
        return service.engine.session_file(action, path)
    except refused_type as error:
        if not retryable(action, error):
            raise
        engine = service.engine
        process = getattr(engine, 'proc', None)
        if process is None or process.poll() is not None:
            raise
        try:
            reclaim(service)
        except (OSError, RuntimeError) as recovery_error:
            print(f'[kurai-session-memory] Reclaim failed; original SAVE refusal retained: {recovery_error}', flush=True)
            raise error from recovery_error
        if service.engine is not engine or engine.proc is not process or process.poll() is not None:
            raise error
        # Physical RAM accounting/page reclamation can lag deallocation. Keep the
        # same live engine and FIFO, and wait instead of immediately failing the shift.
        deadline = time.monotonic() + SETTLE_SECONDS
        last_error = error
        print('[kurai-session-memory] Waiting for SAVE headroom: '
              f'need={required_mib(error)} MiB, limit={SETTLE_SECONDS}s; admission floor unchanged', flush=True)
        for attempt in range(MAX_RETRIES):
            if not wait_for_headroom(service, engine, process, last_error, deadline):
                raise last_error
            try:
                return engine.session_file(action, path)
            except refused_type as failure:
                if not retryable(action, failure):
                    raise
                last_error = failure
                # Unknown telemetry gets one guarded attempt, never a blind loop.
                if required_mib(failure) is None or available_mib() is None:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(0.5, remaining))
        raise last_error
