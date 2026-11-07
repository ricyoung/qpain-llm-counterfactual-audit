"""OpenRouter backend: pinned provider, spend cap, and full capture of every exchange.

Every call records, and returns for the record on disk:
  request       the exact JSON body sent (the Authorization header is never stored)
  raw_response  the complete JSON body returned, unmodified
  http          final status code and response headers
  attempts      one entry per attempt: start time, seconds, status or error, and
                the raw error body when the attempt failed
A call that fails on every attempt raises CallFailed carrying the same detail,
which the runner writes to outputs/panel_errors/ so failures are kept in full.

The key is read at call time from the file named by $OPENROUTER_KEY_FILE (default
~/.config/openrouter_key, mode 0600); it is never logged or stored with a record. Each request
pins one provider with no fallbacks and requires the provider to honour the
parameters sent. A call is refused before it is sent when the ledger total plus a
per-call reserve would exceed the cap.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import os
from pathlib import Path

URL = 'https://openrouter.ai/api/v1/chat/completions'
KEY_FILE = Path(os.environ.get('OPENROUTER_KEY_FILE') or Path.home() / '.config' / 'openrouter_key')
LEDGER = Path(__file__).parent / 'outputs' / 'openrouter_ledger.jsonl'
RESERVE_USD = 0.05
RETRYABLE = (408, 429, 500, 502, 503, 504)
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


class CallFailed(RuntimeError):
    """Every attempt failed; `detail` holds the request and every attempt in full."""
    def __init__(self, message: str, detail: dict):
        super().__init__(message)
        self.detail = detail


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line).get('cost', 0.0) for line in LEDGER.read_text().splitlines() if line.strip())


def _key() -> str:
    return KEY_FILE.read_text().strip()


def build_body(spec: dict, prompt: str, *, seed: int, schema: dict, temperature: float = 0.7) -> dict:
    provider = {'order': [spec['provider']], 'allow_fallbacks': False, 'require_parameters': True}
    if spec.get('quantization'):
        provider['quantizations'] = [spec['quantization']]
    body = {'model': spec['id'], 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature, 'provider': provider, 'usage': {'include': True},
            'response_format': {'type': 'json_schema',
                                'json_schema': {'name': 'answer', 'strict': True, 'schema': schema}},
            'max_tokens': 6000 if spec.get('reasoning') in ('on', 'low', 'medium', 'high') else 600}
    if spec.get('seed', True):
        body['seed'] = seed
    reasoning = spec.get('reasoning')
    if reasoning == 'off':
        body['reasoning'] = {'enabled': False}
    elif reasoning == 'on':
        body['reasoning'] = {'enabled': True}
    elif reasoning in ('low', 'medium', 'high'):
        body['reasoning'] = {'effort': reasoning}
    return body


def call(spec: dict, prompt: str, *, seed: int, schema: dict, cap_usd: float,
         temperature: float = 0.7, timeout: float = 300) -> dict:
    with _lock:
        if spent() + RESERVE_USD > cap_usd:
            raise BudgetExceeded(f'spend cap ${cap_usd:.2f} reached (${spent():.2f} spent)')
    body = build_body(spec, prompt, seed=seed, schema=schema, temperature=temperature)
    payload = json.dumps(body).encode()
    attempts, data, http = [], None, None
    for attempt in range(4):
        started = time.time()
        entry = {'attempt': attempt + 1, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(started))}
        request = urllib.request.Request(URL, data=payload, headers={
            'Authorization': f'Bearer {_key()}', 'Content-Type': 'application/json',
            'X-Title': 'pain-disparity-audit'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode('utf-8', errors='replace')
                http = {'status': response.status, 'headers': dict(response.headers.items())}
            entry.update(seconds=round(time.time() - started, 3), status=http['status'])
            data = json.loads(text)
            attempts.append(entry)
            if 'error' in data:               # HTTP 200 with an error body
                entry['error_body'] = data['error']
                code = (data['error'] or {}).get('code') if isinstance(data['error'], dict) else None
                if code in RETRYABLE and attempt < 3:
                    time.sleep(10 * (attempt + 1)); data = None; continue
                raise CallFailed(f"OpenRouter error: {str(data['error'])[:300]}",
                                 {'request': body, 'attempts': attempts, 'raw_response': data, 'http': http})
            break
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode('utf-8', errors='replace')
            entry.update(seconds=round(time.time() - started, 3), status=exc.code,
                         headers=dict(exc.headers.items()) if exc.headers else {}, error_body=error_text)
            attempts.append(entry)
            if exc.code in RETRYABLE and attempt < 3:
                time.sleep(10 * (attempt + 1)); continue
            raise CallFailed(f'HTTP {exc.code}: {error_text[:300]}', {'request': body, 'attempts': attempts}) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            entry.update(seconds=round(time.time() - started, 3), error=f'{type(exc).__name__}: {exc}')
            attempts.append(entry)
            if attempt < 3:
                time.sleep(10 * (attempt + 1)); continue
            raise CallFailed(f'transfer failed after 4 attempts: {type(exc).__name__}: {exc}',
                             {'request': body, 'attempts': attempts}) from None
    message = data['choices'][0]['message']
    usage = data.get('usage') or {}
    cost = float(usage.get('cost') or 0.0)
    with _lock:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open('a') as handle:
            handle.write(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'id': data.get('id'),
                                     'model': spec['id'], 'reasoning': spec.get('reasoning'),
                                     'provider': data.get('provider'), 'cost': cost,
                                     'prompt_tokens': usage.get('prompt_tokens'),
                                     'completion_tokens': usage.get('completion_tokens'),
                                     'attempts': len(attempts)}) + '\n')
    return {'response': message.get('content') or '', 'thinking': message.get('reasoning') or '',
            'provider': data.get('provider'), 'cost': cost, 'usage': usage,
            'request': body, 'raw_response': data, 'http': http, 'attempts': attempts}
