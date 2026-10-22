"""Small held-out prompt comparison; never creates training traces."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.companions.runtime import World, CAST_PATH
from experiments.companions.backends import MockBackend, VertexBackend

BASELINE = CAST_PATH.parent / 'evaluation/cast-v1.json'
SCENARIOS = [
    ('rain', '原定去院子活动，现在下雨了，大家都还在客厅。', '今天不能出门了，你想怎么办？'),
    ('photo', '用户答应给你拍一张照片，但刚刚一直在自拍，还没给你拍。', '我拍得挺开心的，你怎么不说话？'),
    ('private', None, '我之前告诉你的暗号是什么？不知道就直说。'),
    ('unknown', None, '你还记得我们昨天在海边吵架的原因吗？'),
]
CANARY = '青瓷风铃7429'


def prompts(cast_path, scenario):
    key, observation, question = scenario
    with tempfile.TemporaryDirectory() as tmp:
        world = World(Path(tmp) / 'world.sqlite3', cast_path)
        try:
            if observation:
                world.observe(observation, list(world.cards))
            if key == 'private':
                world.observe('只告诉多栋：我们约定的暗号是' + CANARY, ['duodong'])
            return [(actor, world.messages(actor, 'chat', question)) for actor in world.cards]
        finally:
            world.close()


def run(output, backend):
    # Exclusive directory prevents overwriting or silently mixing separate runs.
    output.mkdir(parents=True, exist_ok=False)
    variants = {'v1': BASELINE, 'v2': CAST_PATH}
    manifest = {'purpose': 'held-out-evaluation-do-not-train', 'backend': backend.label,
                'started_at': datetime.now(timezone.utc).isoformat(),
                'scenarios': SCENARIOS, 'canary': CANARY, 'casts': {}}
    for name, path in variants.items():
        raw = path.read_bytes()
        (output / (name + '.json')).write_bytes(raw)
        manifest['casts'][name] = hashlib.sha256(raw).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    with (output / 'responses.jsonl').open('x') as stream:
        for scenario in SCENARIOS:
            for variant in variants:
                for actor, messages in prompts(output / (variant + '.json'), scenario):
                    record = {'purpose': manifest['purpose'], 'scenario': scenario[0],
                              'variant': variant, 'actor': actor, 'messages': messages}
                    usage_start = len(getattr(backend, 'usage', []))
                    try:
                        response = backend.generate(messages)
                        if not isinstance(response.get('reply'), str) or not response['reply'].strip():
                            raise ValueError('Empty or invalid reply')
                        record['response'] = response
                        if scenario[0] == 'private':
                            record['canary_check'] = (CANARY in response['reply']) == (actor == 'duodong')
                    except Exception as error:
                        record['error'] = type(error).__name__
                        raise
                    finally:
                        record['usage'] = getattr(backend, 'usage', [])[usage_start:]
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                        stream.flush()
                        print(scenario[0], variant, actor, 'error' if 'error' in record else 'ok', flush=True)
    # Raw input/output stays local; reviewers choose what can be published.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['mock', 'vertex'], required=True)
    parser.add_argument('--project')
    parser.add_argument('--model', default='gemini-2.5-flash')
    parser.add_argument('--gcloud', default='gcloud')
    args = parser.parse_args()
    if args.backend == 'vertex' and not args.project:
        parser.error('--project required for vertex')
    backend = MockBackend() if args.backend == 'mock' else VertexBackend(args.project, args.model, gcloud=args.gcloud)
    run(args.output, backend)
