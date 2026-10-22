#!/usr/bin/env python3
"""Five-character CLI; model calls require an explicit backend selection."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.companions.runtime import World, encode
from experiments.companions.backends import MockBackend, UpstreamBackend, VertexBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=ROOT / 'runs/companions/world.sqlite3')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    sub.add_parser('status')
    event = sub.add_parser('observe')
    event.add_argument('--audience', nargs='+', required=True, help='Explicit character IDs, or all')
    event.add_argument('--text', required=True)
    for command in ('step', 'chat'):
        p = sub.add_parser(command)
        p.add_argument('--backend', choices=('mock', 'upstream', 'vertex'), required=True)
        p.add_argument('--config', type=Path)
        p.add_argument('--model')
        p.add_argument('--project')
        p.add_argument('--location', default='global')
        p.add_argument('--gcloud', default='gcloud')
        if command == 'step':
            p.add_argument('--steps', type=int, default=1)
        else:
            p.add_argument('--actor', required=True)
            p.add_argument('--text', required=True)
    replay = sub.add_parser('replay')
    replay.add_argument('--actor', required=True, help='Replay only this character’s visible events')
    sub.add_parser('traces')
    audit = sub.add_parser('audit')
    audit.add_argument('--id', type=int, required=True)
    audit.add_argument('--report', type=Path, help='Completed audit JSON; omit to print template')
    review = sub.add_parser('review')
    review.add_argument('--id', type=int, required=True)
    review.add_argument('--verdict', choices=('accepted', 'rejected'), required=True)
    review.add_argument('--reason', required=True)
    export = sub.add_parser('export')
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'step' and not 1 <= args.steps <= 20:
        parser.error('--steps must be 1–20 (five model calls per step)')
    if getattr(args, 'backend', None) == 'upstream' and (not args.config or not args.model):
        parser.error('upstream requires --config and --model; it can incur API charges')
    if getattr(args, 'backend', None) == 'vertex' and (not args.project or not args.model):
        parser.error('vertex requires --project and --model; it can incur API charges')
    world = World(args.db)
    backend = None
    try:
        if args.command in ('step', 'chat'):
            if args.backend == 'mock':
                backend = MockBackend()
            elif args.backend == 'vertex':
                backend = VertexBackend(args.project, args.model, args.location, args.gcloud)
            else:
                backend = UpstreamBackend(args.config, args.model)
            if args.command == 'step':
                for _ in range(args.steps):
                    print(encode({'tick': world.step(backend), 'backend': backend.label}))
            else:
                print(world.chat(args.actor, args.text, backend))
        elif args.command == 'observe':
            audience = list(world.cards) if args.audience == ['all'] else args.audience
            world.observe(args.text, audience)
            print('Event recorded')
        elif args.command == 'replay':
            for row in world.memories(args.actor):
                print(encode(row))
        elif args.command == 'traces':
            for row in world.db.execute('SELECT * FROM traces ORDER BY id'):
                print(encode(dict(row)))
        elif args.command == 'audit':
            if args.report:
                passed = world.record_audit(args.id, json.loads(args.report.read_text()))
                print(encode({'passed': passed, 'overall_review': 'pending'}))
            else:
                print(encode(world.audit_packet(args.id)))
        elif args.command == 'review':
            world.review(args.id, args.verdict, args.reason)
            print('Review recorded')
        elif args.command == 'export':
            # Exclusive create: never silently overwrite a reviewed dataset.
            rows = list(world.export())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as output:
                for row in rows:
                    output.write(encode(row) + '\n')
            print(encode({'exported': len(rows), 'path': str(args.output)}))
        else:
            print(encode({'tick': world.tick, 'characters': [world.actor(a) for a in world.cards],
                          'note': 'Provisional character cards; not a trained model.'}))
    finally:
        if isinstance(backend, VertexBackend):
            print(encode({'usage': backend.usage}), file=sys.stderr)
        world.close()


if __name__ == '__main__':
    main()
