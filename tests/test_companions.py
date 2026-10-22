import json
import tempfile
import unittest
from pathlib import Path
from experiments.companions.runtime import World
from experiments.companions.backends import MockBackend


class Fixture:
    # Non-mock label exercises production data paths; never used outside temporary tests.
    label = 'test-fixture:not-real-model'

    def __init__(self, callback=None):
        self.callback = callback
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        req = json.loads(messages[-1]['content'])
        if self.callback:
            return self.callback(req)
        if req['task'] == 'chat':
            return {'reply': '我记下了。'}
        return {'action': 'talk', 'utterance': '现在一起休息吧。', 'plan': '私有计划', 'feeling': '平静'}


class CompanionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'world.sqlite3'
        self.world = World(self.path)

    def tearDown(self):
        self.world.close()
        self.tmp.cleanup()

    def test_five_distinct_actors(self):
        self.assertEqual(set(self.world.cards), {'duodong', 'lipu', 'xiuzhen', 'xiaohe', 'zhangtianzhong'})

    def test_private_chat_and_plan_not_in_other_context(self):
        self.world.chat('duodong', '秘密暗号：紫色风筝', Fixture())
        self.world.step(Fixture())
        self.assertIn('紫色风筝', json.dumps(self.world.messages('duodong', 'act'), ensure_ascii=False))
        for actor in set(self.world.cards) - {'duodong'}:
            self.assertNotIn('紫色风筝', json.dumps(self.world.messages(actor, 'act'), ensure_ascii=False))
            self.assertFalse(any(e['actor'] == 'duodong' and e['kind'] == 'reflection'
                                 for e in self.world.memories(actor)))

    def test_public_event_and_user_intervention_reach_all(self):
        self.world.observe('今天下雨，野餐取消。', list(self.world.cards))
        backend = Fixture()
        self.world.step(backend)
        self.assertEqual(len(backend.calls), 5)
        for messages in backend.calls:
            self.assertIn('野餐取消', messages[-1]['content'])

    def test_absent_actor_does_not_hear_speech(self):
        def act(req):
            if req['your_state']['id'] == 'lipu':
                return {'action': 'move', 'destination': '街角', 'plan': '散步', 'feeling': '平静'}
            return {'action': 'talk', 'utterance': '留在客厅说的话', 'plan': '等朋友', 'feeling': '平静'}
        self.world.step(Fixture(act))
        self.assertFalse(any(e['text'] == '留在客厅说的话' for e in self.world.memories('lipu')))
        self.assertTrue(any(e['text'] == '留在客厅说的话' for e in self.world.memories('duodong')))

    def test_restart_preserves_state_and_memory(self):
        self.world.chat('xiaohe', '明天一起散步。', Fixture())
        self.world.step(Fixture())
        expected = self.world.memories('xiaohe')
        self.world.close()
        self.world = World(self.path)
        self.assertEqual(self.world.tick, 1)
        self.assertEqual(self.world.memories('xiaohe'), expected)
        self.assertEqual(self.world.actor('xiaohe')['plan'], '私有计划')

    def test_failure_of_fifth_call_rolls_back_entire_step(self):
        def act(req):
            if req['your_state']['id'] == 'zhangtianzhong':
                raise RuntimeError('provider unavailable')
            return {'action': 'rest', 'plan': '休息', 'feeling': '平静'}
        with self.assertRaises(RuntimeError):
            self.world.step(Fixture(act))
        self.assertEqual(self.world.version(), (0, 0, 0))

    def test_invalid_action_does_not_change_state(self):
        with self.assertRaises(ValueError):
            self.world.step(Fixture(lambda _: {'action': 'move', 'destination': '不存在',
                                                'plan': '出门', 'feeling': '平静'}))
        self.assertEqual(self.world.version(), (0, 0, 0))

    def test_concurrent_observation_invalidates_generation(self):
        second = World(self.path)
        try:
            changed = False
            def act(req):
                nonlocal changed
                if not changed:
                    second.observe('门铃响了。', ['duodong'])
                    changed = True
                return {'action': 'rest', 'plan': '休息', 'feeling': '平静'}
            with self.assertRaisesRegex(RuntimeError, 'World changed'):
                self.world.step(Fixture(act))
            self.assertEqual(self.world.tick, 0)
            self.assertEqual(len(self.world.memories('duodong')), 1)
        finally:
            second.close()

    def test_mock_never_becomes_training_data(self):
        self.world.step(MockBackend())
        with self.assertRaises(ValueError):
            self.world.review(1, 'accepted', '不应接受离线占位数据')
        self.assertEqual(list(self.world.export()), [])
        with self.assertRaisesRegex(ValueError, 'separate databases'):
            self.world.step(Fixture())

    def test_export_only_reviewed_examples_and_original_context(self):
        self.world.chat('duodong', '你好', Fixture())
        self.assertEqual(list(self.world.export()), [])
        report = self.world.audit_packet(1)
        report['reviewer'] = 'test-reviewer'
        report['fields']['reply'].update(verdict='nonfactual', reason='当前回应，无往事断言')
        self.world.record_audit(1, report)
        self.world.review(1, 'accepted', '角色身份和记忆一致')
        self.world.observe('未来发生的事', ['duodong'])
        rows = list(self.world.export())
        self.assertEqual(len(rows), 1)
        self.assertNotIn('未来发生的事', json.dumps(rows, ensure_ascii=False))
        self.assertEqual(rows[0]['metadata']['actor'], 'duodong')
        self.world.review(1, 'rejected', '复核后排除')
        self.assertEqual(list(self.world.export()), [])

    def test_unknown_audience_fails_without_event(self):
        with self.assertRaises(ValueError):
            self.world.observe('消息', ['unknown'])
        self.assertEqual(self.world.version(), (0, 0, 0))

    def test_chat_failure_leaves_no_half_message(self):
        with self.assertRaises(ValueError):
            self.world.chat('duodong', '你好', Fixture(lambda _: {'reply': ''}))
        self.assertEqual(self.world.version(), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
