import copy
import tempfile
import unittest
from pathlib import Path
from experiments.companions.runtime import World


class Backend:
    label = 'test-fixture'
    def generate(self, messages):
        return {'reply': '门还没修好。'}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.world = World(Path(self.tmp.name) / 'world.sqlite3')
        self.world.observe('院门还没修好。', ['duodong'])
        self.world.chat('duodong', '门修好了吗？', Backend())
        self.report = self.world.audit_packet(1)
        self.report['reviewer'] = 'test-reviewer'
        self.report['fields']['reply'].update(verdict='supported', evidence=['event:1'], reason='观察明确说明门未修好')

    def tearDown(self):
        self.world.close()
        self.tmp.cleanup()

    def test_gate_and_reaudit_revokes_approval(self):
        with self.assertRaises(ValueError):
            self.world.review(1, 'accepted', '旧式接受不足以导出')
        self.world.record_audit(1, self.report)
        self.assertEqual(list(self.world.export()), [])
        self.world.review(1, 'accepted', '综合审核通过')
        self.assertEqual(len(list(self.world.export())), 1)
        self.report['fields']['reply'].update(verdict='uncertain', problem_quote='门还没修好', reason='复核待查')
        self.assertFalse(self.world.record_audit(1, self.report))
        self.assertEqual(list(self.world.export()), [])
        self.assertEqual(self.world.db.execute('SELECT count(*) FROM fact_audits').fetchone()[0], 2)

    def test_future_and_private_evidence_unavailable(self):
        self.world.observe('以后门修好了。', ['duodong'])
        self.world.observe('其他角色的秘密。', ['lipu'])
        for ref in ('event:4', 'event:5'):
            report = copy.deepcopy(self.report)
            report['fields']['reply']['evidence'] = [ref]
            with self.assertRaises(ValueError):
                self.world.record_audit(1, report)

    def test_missing_fields_and_changed_text_fail_closed(self):
        for change in ('fields', 'quote', 'fingerprint', 'evidence'):
            report = copy.deepcopy(self.report)
            if change == 'fields': report['fields'] = {}
            elif change == 'fingerprint': report['fingerprint'] = 'stale'
            else: report['fields']['reply'][change] = [] if change == 'evidence' else '不同文本'
            with self.assertRaises(ValueError):
                self.world.record_audit(1, report)

    def test_legacy_acceptance_and_modified_trace_not_exported(self):
        self.world.db.execute("UPDATE traces SET review='accepted' WHERE id=1")
        self.world.db.commit()
        self.assertEqual(list(self.world.export()), [])
        self.world.record_audit(1, self.report)
        self.world.review(1, 'accepted', '通过')
        self.world.db.execute("UPDATE traces SET response=? WHERE id=1", ('{"reply":"我昨天去了海边。"}',))
        self.world.db.commit()
        self.assertEqual(list(self.world.export()), [])

    def test_reviewer_cannot_replace_authoritative_evidence(self):
        self.report['evidence'] = {'event:1': '恶意替换的文本'}
        self.world.record_audit(1, self.report)
        self.world.review(1, 'accepted', '通过')
        result = list(self.world.export())[0]
        self.assertEqual(result['metadata']['fact_audit']['evidence']['event:1']['text'], '院门还没修好。')
