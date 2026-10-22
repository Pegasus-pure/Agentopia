"""Structured reviewer decisions, not an automatic truth detector."""
import hashlib
import json


def fingerprint(trace):
    value = {k: trace[k] for k in ('id', 'tick', 'actor', 'backend', 'messages', 'response')}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def packet(trace):
    messages = json.loads(trace['messages'])
    request = json.loads(messages[-1]['content'])
    response = json.loads(trace['response'])
    evidence = {'current_state': request['your_state'],
                'system': [m['content'] for m in messages if m['role'] == 'system']}
    if 'user_message' in request:
        evidence['user_message'] = request['user_message']
    for event in request['visible_memories']:
        evidence[f"event:{event['id']}"] = event
    # Include every output field, including action/plan/feeling: no unchecked prose.
    return {'trace_id': trace['id'], 'fingerprint': fingerprint(trace),
            'reviewer': '', 'evidence': evidence, 'response': response,
            'fields': {key: {'quote': value, 'verdict': 'undecided', 'evidence': [],
                             'reason': '', 'problem_quote': ''} for key, value in response.items()}}


def validate(trace, report):
    expected = packet(trace)
    if report.get('trace_id') != trace['id'] or report.get('fingerprint') != expected['fingerprint']:
        raise ValueError('Audit is for a different or changed trace')
    if not isinstance(report.get('reviewer'), str) or not report['reviewer'].strip():
        raise ValueError('Reviewer identity required')
    fields = report.get('fields')
    if not isinstance(fields, dict) or set(fields) != set(expected['fields']):
        raise ValueError('Audit must cover every response field')
    passed = True
    for key, item in fields.items():
        if not isinstance(item, dict) or item.get('quote') != expected['response'][key]:
            raise ValueError('Audit quote must match complete response field')
        verdict = item.get('verdict')
        if verdict not in ('supported', 'nonfactual', 'unsupported', 'uncertain'):
            raise ValueError('Every field needs an explicit verdict')
        if not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError('Every field needs an explanation')
        refs = item.get('evidence')
        if not isinstance(refs, list) or any(not isinstance(r, str) or r not in expected['evidence'] for r in refs):
            raise ValueError('Evidence must refer to original visible input')
        if verdict == 'supported' and not refs:
            raise ValueError('Supported facts need evidence references')
        if verdict in ('unsupported', 'uncertain'):
            quote = item.get('problem_quote')
            if not isinstance(quote, str) or not quote or not isinstance(item['quote'], str) or quote not in item['quote']:
                raise ValueError('Flag the exact unsupported or uncertain text')
            passed = False
    # Discard edited evidence/response copies; retain authoritative originals.
    expected.update(reviewer=report['reviewer'], fields=fields)
    return passed, expected
