#!/usr/bin/env python3
"""Compare quad centroids, sorted triangle indices and cached viewpoints with vanilla 26.3."""
import argparse
from collections import defaultdict
import json
import os
import random
import struct
import subprocess

try:
    from . import lab, test_rust_parity as base, test_model_baking_parity as baking
except ImportError:
    import lab, test_rust_parity as base, test_model_baking_parity as baking

COMPACT_MIN = -8.0
COMPACT_MAX = 24.0 - 1.0 / 2048.0


def quad_at(cx, cy, cz, hx, hy, hz):
    return [
        [cx - hx, cy - hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx + hx, cy + hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
    ]


def boundary_cameras(origin, rng, extra=()):
    cameras = [[o + d for o in origin] for d in [-32.5, -32.1, -31.9, -16, -0.1, 0, 0.9, 1.1, 15.9, 16, 32, 48]]
    cameras.extend(extra)
    cameras.extend([origin[i] + rng.uniform(-64, 64) for i in range(3)] for _ in range(8))
    return cameras


def cases():
    rng = random.Random(563)
    # Equal distances and a skewed quad detect unstable ties and averaging all four corners.
    yield ('tie_and_skewed', 'exact', {
        'origin': [0, 0, 0],
        'quads': [
            [[-1, 0, 0]] * 4,
            [[1, 0, 0]] * 4,
            [[0, 0, 0], [100, 100, 100], [2, 2, 2], [200, 200, 200]],
        ],
        'cameras': [[0, 0, 0], [0.1, 0.1, 0.1], [1, 0, 0], [-1, 0, 0]],
    })
    for idx, count in enumerate([0, 1, 2, 64, *[rng.randrange(3, 100) for _ in range(200)]]):
        origin = [rng.choice([-30000000, -64, -16, 0, 16, 30000000]) for _ in range(3)]
        # SectionPos packs Y into fewer bits; world-border coordinates apply to X/Z.
        origin[1] = rng.choice([-64, -16, 0, 16, 304])
        quads = [[[rng.uniform(-1, 17) for _ in range(3)] for _ in range(4)] for _ in range(count)]
        # Walk through block and section boundaries, including negative and diagonal viewpoints.
        cameras = [[o + d for o in origin] for d in [-32.5, -32.1, -31.9, -16, -0.1, 0, 0.9, 1.1, 15.9, 16, 32, 48]]
        cameras += [[origin[i] + rng.uniform(-64, 64) for i in range(3)] for _ in range(8)]
        yield (f'random_{idx}_{count}', 'exact', {'origin': origin, 'quads': quads, 'cameras': cameras})

    # Targeted storage and threshold cases exercising Compact (1/16 and 1/2048), Full (>=128 radix),
    # vertex-level Coplanar rejection (sloped/vertical), and Coplanar flat horizontal planes.
    extra_rng = random.Random(564)

    # 1/16-grid centers (< 128 packed u64 path) with exact symmetric and co-located ties.
    origin = [16, 0, -16]
    quads = []
    for g in range(16):
        dx = (g % 4) + 1.0 / 16.0
        dy = (g // 4) + 3.0 / 16.0
        dz = ((g * 3) % 5) + 0.5
        for sx, sy, sz in [(1, 1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1)]:
            quads.append(quad_at(8.0 + sx * dx, 4.0 + sy * dy, 8.0 + sz * dz, 1.0 / 16.0, 1.0 / 16.0, 0.0))
    for p in range(16):
        cx = (p % 8) - 2.0 + ((p % 15) + 1) / 16.0
        cy = (p // 8) + 7.0 / 16.0
        cz = ((p * 5) % 12) + 9.0 / 16.0
        q = quad_at(cx, cy, cz, 1.0 / 16.0, 1.0 / 16.0, 0.0)
        quads.extend([q, q])
    yield ('compact_1_16_packed', 'compact_1_16', {
        'origin': origin,
        'quads': quads,
        'cameras': boundary_cameras(origin, extra_rng, [
            [origin[0] + 8.0, origin[1] + 4.0, origin[2] + 8.0],
            [origin[0] + 0.1, origin[1] + 0.0, origin[2] + 0.0],
        ]),
    })

    # 1/16-grid centers (>= 128 LSD radix path) at world-border coordinates.
    origin = [-30000000, 64, 30000000]
    quads = []
    for i in range(256):
        cx = (i % 16) + ((i % 7) + 1) / 16.0
        cy = ((i // 16) % 4) + ((i % 5) + 1) / 16.0
        cz = ((i // 64) % 4) * 3.0 + ((i % 3) + 1) / 16.0
        quads.append(quad_at(cx, cy, cz, 1.0 / 16.0, 1.0 / 16.0, 0.0))
    yield ('compact_1_16_radix', 'compact_1_16', {
        'origin': origin,
        'quads': quads,
        'cameras': boundary_cameras(origin, extra_rng, [
            [origin[0] + 8.0, origin[1] + 2.0, origin[2] + 8.0],
            [origin[0] + 0.1, origin[1] + 0.0, origin[2] + 0.0],
        ]),
    })

    # 1/2048-grid centers (< 128 packed u64 path) with odd 1/2048 steps, range endpoints, and ties.
    origin = [0, -64, 0]
    quads = []
    for g in range(16):
        dx = (g % 4) + (2 * (g % 3) + 1) / 2048.0
        dy = (g // 4) + (2 * (g % 5) + 1) / 2048.0
        dz = ((g * 3) % 5) + 1.0 / 16.0
        for sx, sy, sz in [(1, 1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1)]:
            quads.append(quad_at(8.0 + sx * dx, 4.0 + sy * dy, 8.0 + sz * dz, 1.0 / 2048.0, 1.0 / 16.0, 1.0 / 2048.0))
    # Range endpoints [-8.0, 24.0 - 1/2048] and adjacent 1/2048 pairs.
    edge_centers = [
        (-8.0, 0.0 + 1.0 / 2048.0, 1.0),
        (-8.0 + 1.0 / 2048.0, 0.0 + 1.0 / 2048.0, 1.0),
        (24.0 - 2.0 / 2048.0, 3.0 + 3.0 / 2048.0, 5.0),
        (24.0 - 1.0 / 2048.0, 3.0 + 3.0 / 2048.0, 5.0),
    ]
    for cx, cy, cz in edge_centers:
        q = quad_at(cx, cy, cz, 1.0 / 2048.0, 1.0 / 16.0, 1.0 / 2048.0)
        quads.extend([q, q])
    for p in range(12):
        cx = (p % 6) + (2 * p + 1) / 2048.0
        cy = (p // 6) + 1.0 + (2 * p + 3) / 2048.0
        cz = ((p * 5) % 8) + 1.0 / 2048.0
        q = quad_at(cx, cy, cz, 1.0 / 2048.0, 1.0 / 16.0, 1.0 / 2048.0)
        quads.extend([q, q])
    yield ('compact_1_2048_packed', 'compact_1_2048', {
        'origin': origin,
        'quads': quads,
        'cameras': boundary_cameras(origin, extra_rng, [
            [origin[0] + 8.0, origin[1] + 4.0, origin[2] + 8.0],
            [origin[0] + 0.1, origin[1] + 0.0, origin[2] + 0.0],
            [origin[0] - 7.9, origin[1] + 0.0, origin[2] + 1.0],
            [origin[0] + 23.9, origin[1] + 3.0, origin[2] + 5.0],
        ]),
    })

    # 1/2048-grid centers (>= 128 LSD radix path) with odd 1/2048 steps across multiple Y layers.
    origin = [0, 0, 0]
    quads = []
    for i in range(256):
        cx = (i % 16) + 0.5 + (1.0 / 2048.0 if i % 2 else 0.0)
        cy = ((i // 16) % 4) + (1.0 / 2048.0 if i % 7 == 0 else (1.0 / 16.0 if i % 7 == 1 else 0.5))
        cz = ((i // 64) % 4) * 2.0 + 0.5
        quads.append(quad_at(cx, cy, cz, 1.0 / 2048.0, 1.0 / 16.0, 1.0 / 2048.0))
    yield ('compact_1_2048_radix', 'compact_1_2048', {
        'origin': origin,
        'quads': quads,
        'cameras': boundary_cameras(origin, extra_rng, [
            [7.25, 2.75, 3.125],
            [0.1, 0.0, 0.0],
        ]),
    })

    # Sloped flowing-water tops sharing center y = 0.7 (>= 128 radix, Centers::Full, must NOT be Coplanar).
    origin = [0, 48, 0]
    quads = []
    for z in range(10):
        for x in range(16):
            quads.append([
                [float(x), 0.8, float(z)],
                [float(x), 0.8, float(z + 1)],
                [float(x + 1), 0.6, float(z + 1)],
                [float(x + 1), 0.6, float(z)],
            ])
    yield ('full_sloped_water_radix', 'exact', {
        'origin': origin,
        'quads': quads,
        'cameras': boundary_cameras(origin, extra_rng, [
            [8.5, 48.7, -4.0],
            [8.5, 48.7, 20.0],
            [0.5, 48.7, 5.0],
        ]),
    })

    # Vertical faces sharing center y = 0.5 (Centers::Compact, must NOT be Coplanar).
    origin = [0, 0, 0]
    quads = []
    for z in range(4):
        for x in range(8):
            quads.append([
                [float(x), 0.0, float(z)],
                [float(x), 1.0, float(z)],
                [float(x + 1), 1.0, float(z)],
                [float(x + 1), 0.0, float(z)],
            ])
    yield ('compact_vertical_same_y', 'exact', {
        'origin': origin,
        'quads': quads,
        'cameras': boundary_cameras(origin, extra_rng, [[4.5, 0.5, -2.0], [4.5, 0.5, 6.0]]),
    })

    # Single flat horizontal ocean plane (Centers::Coplanar): 16x16 top + co-located underside.
    origin = [0, 48, 0]
    y_ocean = 14.887889
    quads = []
    for z in range(16):
        for x in range(16):
            fx, fz = float(x), float(z)
            top = [[fx, y_ocean, fz], [fx, y_ocean, fz + 1.0], [fx + 1.0, y_ocean, fz + 1.0], [fx + 1.0, y_ocean, fz]]
            back = [top[1], top[0], top[3], top[2]]
            quads.extend([top, back])
    yield ('coplanar_ocean_plane', 'coplanar', {
        'origin': origin,
        'quads': quads,
        'cameras': boundary_cameras(origin, extra_rng, [
            [8.5, 70.0, 8.5],
            [9.5, 70.0, 8.5],
            [8.5, 62.9, 8.5],
            [8.5, 50.0, 8.5],
            [0.5, 50.0, 0.5],
        ]),
    })

    # Single flat horizontal plane at y = 0.5 (Centers::Coplanar via vertex-level from_quads).
    origin = [0, 16, 0]
    quads = [
        [[float(x), 0.5, float(z)], [float(x), 0.5, float(z + 1)],
         [float(x + 1), 0.5, float(z + 1)], [float(x + 1), 0.5, float(z)]]
        for z in range(8) for x in range(8)
    ]
    yield ('coplanar_half_slab_plane', 'coplanar', {
        'origin': origin,
        'quads': quads,
        'cameras': boundary_cameras(origin, extra_rng, [[4.5, 20.0, 4.5], [4.5, 12.0, 4.5]]),
    })


def unpack_f32(bits):
    return struct.unpack('<f', struct.pack('<i', bits))[0]


def verify_compact_grid(name, centers_bits, denom, require_odd=False):
    ys = set()
    saw_step = False
    for c in centers_bits:
        coords = [unpack_f32(b) for b in c]
        ys.add(coords[1])
        for v in coords:
            assert COMPACT_MIN <= v <= COMPACT_MAX, f'{name}: coordinate {v} outside Compact range'
            scaled = v * denom
            assert scaled == int(scaled), f'{name}: coordinate {v} is not on 1/{denom} grid'
            iv = int(scaled)
            if require_odd:
                if iv % 2 != 0:
                    saw_step = True
            elif iv % denom != 0:
                saw_step = True
    assert len(ys) > 1, f'{name}: compact case should span multiple Y layers'
    assert saw_step, f'{name}: expected non-trivial 1/{denom} step coordinates'


def quad_order(indices, quad_count, label):
    assert len(indices) == quad_count * 6, f'{label}: expected {quad_count * 6} indices, got {len(indices)}'
    order = []
    for k in range(quad_count):
        base_idx = indices[k * 6]
        assert base_idx % 4 == 0, f'{label}: unaligned quad base {base_idx}'
        expected_quad = [base_idx, base_idx + 1, base_idx + 2, base_idx + 2, base_idx + 3, base_idx]
        assert indices[k * 6:(k + 1) * 6] == expected_quad, f'{label}: bad triangle winding at quad slot {k}'
        order.append(base_idx // 4)
    assert sorted(order) == list(range(quad_count)), f'{label}: indices are not a permutation of 0..{quad_count}'
    return order


def group_by_center(order, center_keys):
    groups = defaultdict(list)
    for q in order:
        groups[center_keys[q]].append(q)
    return groups


def verify_coplanar_case(i, name, spec, exp, act, build):
    quads = spec['quads']
    assert quads, f'case {i} ({name}): coplanar case must be non-empty'
    y0 = quads[0][0][1]
    assert 0.0 <= y0 <= 16.0, f'case {i} ({name}): plane y={y0} outside section'
    for q in quads:
        xs = [v[0] for v in q]
        zs = [v[2] for v in q]
        assert all(v[1] == y0 for v in q), f'case {i} ({name}): non-flat quad in coplanar case'
        assert max(xs) - min(xs) <= 1.0 and max(zs) - min(zs) <= 1.0, f'case {i} ({name}): quad spans multiple cells'
        assert 0.0 <= min(xs) and max(xs) <= 16.0 and 0.0 <= min(zs) and max(zs) <= 16.0

    # Coplanar intentionally uploads once on frame 0 and skips all subsequent re-sorts.
    assert act['frames'][0]['update'] is True, f'case {i} ({name}): first frame must upload; inspect {build}'
    assert all(not f['update'] for f in act['frames'][1:]), (
        f'case {i} ({name}): coplanar case should skip re-sorting after frame 0; inspect {build}'
    )
    assert any(f['update'] for f in exp['frames'][1:]), (
        f'case {i} ({name}): camera path did not trigger vanilla re-sorts'
    )
    first_indices = act['frames'][0]['indices']
    assert all(f['indices'] == first_indices for f in act['frames'][1:]), (
        f'case {i} ({name}): coplanar cached indices changed without an update; inspect {build}'
    )

    quad_count = len(quads)
    center_keys = [tuple(c) for c in exp['centers']]
    act_order = quad_order(first_indices, quad_count, f'case {i} ({name}) rust')
    act_groups = group_by_center(act_order, center_keys)
    for fi, exp_frame in enumerate(exp['frames']):
        exp_order = quad_order(exp_frame['indices'], quad_count, f'case {i} ({name}) vanilla frame {fi}')
        exp_groups = group_by_center(exp_order, center_keys)
        assert act_groups == exp_groups, (
            f'case {i} ({name}) frame {fi}: co-located quad draw order differs from vanilla; inspect {build}'
        )


def verify_results(case_specs, expected, actual, build):
    assert len(expected) == len(actual) == len(case_specs), 'case count differs'
    coplanar_count = 0
    for i, ((name, kind, spec), exp, act) in enumerate(zip(case_specs, expected, actual)):
        assert act['centers'] == exp['centers'], f'centroid mismatch in case {i} ({name}); inspect {build}'
        assert len(act['frames']) == len(exp['frames']), f'frame count mismatch in case {i} ({name}); inspect {build}'
        if kind == 'compact_1_16':
            verify_compact_grid(name, act['centers'], 16, require_odd=False)
        elif kind == 'compact_1_2048':
            verify_compact_grid(name, act['centers'], 2048, require_odd=True)
        if kind == 'coplanar':
            coplanar_count += 1
            verify_coplanar_case(i, name, spec, exp, act, build)
            continue
        vanilla_cached = None
        for fi, (exp_f, act_f) in enumerate(zip(exp['frames'], act['frames'])):
            assert act_f['update'] == exp_f['update'], (
                f'update flag mismatch in case {i} ({name}) frame {fi}; inspect {build}'
            )
            if exp_f['update']:
                vanilla_cached = exp_f['indices']
            assert act_f['indices'] == vanilla_cached, (
                f'sorting mismatch in case {i} ({name}) frame {fi}; inspect {build}'
            )
        assert vanilla_cached is not None, f'case {i} ({name}): expected at least one update frame'
    return coplanar_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='fast', help='Cargo build profile')
    args = parser.parse_args()
    lock, jars, _ = base.verify_artifacts()
    build = lab.LOCAL / 'translucency-parity'
    classes = build / 'classes'
    classes.mkdir(parents=True, exist_ok=True)
    case_specs = list(cases())
    inputs = build / 'inputs.jsonl'
    inputs.write_text(''.join(json.dumps(spec) + '\n' for _, _, spec in case_specs), encoding='utf-8')
    dependencies = os.pathsep.join(map(str, [jars['client'], *baking.client_libraries()]))

    def run(command, name):
        with (build / (name + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, command)), cwd=lab.ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        if result.returncode:
            raise RuntimeError((build / (name + '.log')).read_text()[-4000:])

    run(['javac', '--release', lock['minecraft']['java_major'], '-proc:none', '-cp', dependencies, '-d', classes,
         lab.ROOT / 'mods/horizon/src/test/java/dev/horizonlab/parity/VanillaTranslucencyOracle.java'], 'compile-java')
    vanilla, rust = build / 'vanilla.jsonl', build / 'rust.jsonl'
    run(['java', '-cp', str(classes) + os.pathsep + dependencies, 'dev.horizonlab.parity.VanillaTranslucencyOracle', inputs, vanilla], 'vanilla')
    run(['cargo', 'run', '--quiet', '--locked', '--profile', args.profile, '-p', 'horizon-client-lib', '--example', 'client_probe', '--', 'translucency_fixture', inputs, rust], 'rust')
    expected = [json.loads(line) for line in vanilla.read_text().splitlines()]
    actual = [json.loads(line) for line in rust.read_text().splitlines()]
    coplanar_count = verify_results(case_specs, expected, actual, build)
    frames = sum(len(c['frames']) for c in expected)
    print(
        f'translucency parity ok: {len(expected)} cases ({coplanar_count} coplanar), {frames} camera views; '
        'centroids, sorted indices, coplanar draw order, and cache decisions match'
    )


if __name__ == '__main__':
    main()
