import io
import json
import subprocess
import unittest
from unittest.mock import patch
from experiments.companions.backends import VertexBackend


class VertexTests(unittest.TestCase):
    def test_system_role_json_and_no_thought_in_result(self):
        body = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
            {'thought': True, 'text': 'hidden'}, {'text': '{"reply":"你好"}'}]}}],
                'usageMetadata': {'totalTokenCount': 12}}
        backend = VertexBackend('example-project', 'gemini-2.5-flash')
        auth = subprocess.CompletedProcess([], 0, stdout='test-token\n', stderr='')
        with patch('subprocess.run', return_value=auth), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(body).encode())) as send:
            result = backend.generate([{'role': 'system', 'content': '角色'}, {'role': 'user', 'content': '{"task":"chat","user_message":"你好"}'}])
            request = send.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(payload['systemInstruction']['parts'][0]['text'], '角色')
            self.assertNotIn('test-token', request.data.decode())
            self.assertEqual(payload['generationConfig']['responseMimeType'], 'application/json')
            self.assertEqual(payload['generationConfig']['responseSchema']['required'], ['reply'])
            self.assertEqual(result, {'reply': '你好'})
            self.assertEqual(backend.usage, [{'totalTokenCount': 12}])

    def test_truncated_generation_rejected(self):
        body = {'candidates': [{'finishReason': 'MAX_TOKENS', 'content': {'parts': [{'text': '{"reply":"你好"}'}]}}]}
        with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, stdout='test-token')), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(body).encode())):
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                VertexBackend('project', 'model').generate([{'role': 'user', 'content': '{"task":"chat"}'}])

    def test_auth_failure_never_sends_request(self):
        with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 1, stdout='', stderr='private details')), patch('urllib.request.urlopen') as send:
            with self.assertRaisesRegex(RuntimeError, 'access token'):
                VertexBackend('project', 'model').generate([])
            send.assert_not_called()

    def test_invalid_resource_identifier_rejected(self):
        with self.assertRaises(ValueError):
            VertexBackend('project', '../model')
