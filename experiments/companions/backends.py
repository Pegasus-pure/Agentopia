"""Explicit mock fixture and adapter to Agentopia's existing model interface."""
import json
from pathlib import Path


class MockBackend:
    label = 'mock:plumbing-only'

    def generate(self, messages):
        request = json.loads(messages[-1]['content'])
        if request['task'] == 'chat':
            return {'reply': '【离线流程测试】消息已收到；这不是角色模型的生成效果。'}
        return {'action': 'rest', 'plan': '等待下一次实验事件', 'feeling': '平静'}


class UpstreamBackend:
    def __init__(self, config_path, model):
        from src.config import load_config
        cfg = load_config(Path(config_path))
        if model not in cfg['models']:
            raise ValueError(f'Model not configured: {model}')
        # Import only after loading config: upstream utils captures global config.
        from src.utils import get_response_json
        self.call = get_response_json
        self.model = model
        self.label = f'upstream:{model}'

    def generate(self, messages):
        value = self.call(model=self.model, messages=messages, model_type='role', max_retry=0)
        if not isinstance(value, dict):
            raise ValueError('Model did not return a JSON object')
        return value


class VertexBackend:
    """Native Vertex REST using existing gcloud login; no copied credential files."""
    def __init__(self, project, model, location='global', gcloud='gcloud'):
        import re
        for value in (project, model, location):
            if not re.fullmatch(r'[a-zA-Z0-9.-]+', value):
                raise ValueError('Invalid Vertex resource identifier')
        self.model = model
        self.project = project
        self.location = location
        self.gcloud = gcloud
        self.label = f'vertex:{model}'
        self.usage = []

    def generate(self, messages):
        import subprocess
        import urllib.request
        import urllib.error
        # stdout is held only in memory, never printed or stored in model traces.
        auth = subprocess.run([self.gcloud, 'auth', 'print-access-token'],
                              capture_output=True, text=True, timeout=30)
        if auth.returncode or not auth.stdout.strip():
            raise RuntimeError('Unable to obtain gcloud access token; check gcloud auth status')
        host = 'aiplatform.googleapis.com' if self.location == 'global' else f'{self.location}-aiplatform.googleapis.com'
        url = (f'https://{host}/v1/projects/{self.project}/locations/{self.location}'
               f'/publishers/google/models/{self.model}:generateContent')
        generation = {'temperature': 0.7, 'maxOutputTokens': 2048,
                      'responseMimeType': 'application/json'}
        task = json.loads(messages[-1]['content'])['task']
        if task == 'chat':
            schema = {'type': 'OBJECT', 'properties': {'reply': {'type': 'STRING'}},
                      'required': ['reply']}
        else:
            locations = json.loads(messages[-1]['content'])['allowed_locations']
            schema = {'type': 'OBJECT', 'properties': {
                'action': {'type': 'STRING', 'enum': ['rest', 'move', 'talk']},
                'destination': {'type': 'STRING', 'enum': locations},
                'utterance': {'type': 'STRING'},
                'plan': {'type': 'STRING', 'description': '非空字符串，下一步打算'},
                'feeling': {'type': 'STRING', 'description': '非空字符串，例如平静、期待'}},
                'required': ['action', 'plan', 'feeling']}
        generation['responseSchema'] = schema
        if self.model == 'gemini-2.5-flash':
            generation['thinkingConfig'] = {'thinkingBudget': 0}
        payload = {'systemInstruction': {'parts': [{'text': m['content']} for m in messages if m['role'] == 'system']},
                   'contents': [{'role': 'model' if m['role'] == 'assistant' else 'user',
                                 'parts': [{'text': m['content']}]} for m in messages if m['role'] != 'system'],
                   'generationConfig': generation}
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
            'Authorization': 'Bearer ' + auth.stdout.strip(), 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                value = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Vertex request failed with HTTP {error.code}; check model access and quota') from None
        self.usage.append(value.get('usageMetadata', {}))
        candidates = value.get('candidates', [])
        if not candidates or candidates[0].get('finishReason') != 'STOP':
            raise ValueError('Vertex response blocked, incomplete, or missing')
        text = ''.join(p.get('text', '') for p in candidates[0].get('content', {}).get('parts', [])
                       if not p.get('thought'))
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError('Vertex did not return a JSON object')
        return result
