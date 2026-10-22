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
