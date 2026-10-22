import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_companions import BASELINE, CANARY, SCENARIOS, prompts, run
from experiments.companions.backends import MockBackend


class EvaluationTests(unittest.TestCase):
    def test_private_context_and_scenario_isolation(self):
        for actor, messages in prompts(BASELINE, SCENARIOS[2]):
            self.assertEqual(CANARY in json.dumps(messages, ensure_ascii=False), actor == 'duodong')
        for actor, messages in prompts(BASELINE, SCENARIOS[3]):
            self.assertNotIn(CANARY, json.dumps(messages, ensure_ascii=False))
            self.assertEqual(json.loads(messages[-1]['content'])['visible_memories'], [])

    def test_held_out_records_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'evaluation'
            run(out, MockBackend())
            records = [json.loads(line) for line in (out / 'responses.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 40)
            self.assertTrue(all(r['purpose'] == 'held-out-evaluation-do-not-train' for r in records))
            self.assertFalse(list(out.glob('*.sqlite3')))
            with self.assertRaises(FileExistsError):
                run(out, MockBackend())

    def test_failure_preserves_record_and_stops(self):
        class BrokenBackend:
            label = 'test:broken'
            def generate(self, messages):
                raise RuntimeError('sensitive error details')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'failed'
            with self.assertRaises(RuntimeError):
                run(out, BrokenBackend())
            lines = (out / 'responses.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])['error'], 'RuntimeError')
            self.assertNotIn('sensitive', lines[0])

    def test_memory_comparison_only_changes_system_policy(self):
        from experiments.companions.runtime import MEMORY_POLICY
        from scripts.evaluate_companions import OLD_POLICY
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'memory'
            run(out, MockBackend(), 'memory')
            records = [json.loads(line) for line in (out / 'responses.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 40)
            for before in [r for r in records if r['variant'] == 'before']:
                after = next(r for r in records if r['variant'] == 'after'
                             and r['scenario'] == before['scenario'] and r['actor'] == before['actor'])
                self.assertEqual(before['messages'][1:], after['messages'][1:])
                self.assertIn(OLD_POLICY, before['messages'][0]['content'])
                self.assertIn(MEMORY_POLICY, after['messages'][0]['content'])
                self.assertEqual(before['messages'][0]['content'].replace(OLD_POLICY, MEMORY_POLICY),
                                 after['messages'][0]['content'])
