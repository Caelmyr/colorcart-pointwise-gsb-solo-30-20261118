"""自动隔色配色：只在当前色板内选色，相邻区域不同色，锁定区域保持原色。

预览完全只读；采纳时写入一次编辑检查点，因而可撤销/重做。方案带有区域图
指纹，预览后若区域、色板、锁色或当前颜色发生变化，旧方案不会被采纳。
"""
import hashlib
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from image_lab.validation import Conflict, Invalid

from .core import checkpoint, png, render, rgb

MODES = ('tone', 'inks')
MODE_LABELS = {'tone': '保留色调', 'inks': '省印色'}
MAX_REGIONS = 16
MAX_COLORS = 6
STALE = '方案已过期：预览后区域、色板、锁色或颜色发生了变化，请重新生成。'


def _lab(colors):
    """sRGB 十六进制色转换为 CIELAB（D65），用于计算 CIE76 色差。"""
    a = np.asarray(colors, float) / 255.0
    a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = a @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def _fingerprint(p):
    """区域几何、名称、锁色、原色与当前色、色板的摘要。"""
    regions = sorted((int(k), str(r.get('name', '')), bool(r['locked']), r['source_color'].lower(), r['color'].lower()) for k, r in p['regions'].items())
    labels = np.ascontiguousarray(np.asarray(p['labels'], np.int32))
    digest = hashlib.sha256()
    digest.update(json.dumps([list(labels.shape), regions, [c.lower() for c in p['palette']]], sort_keys=True).encode())
    digest.update(labels.tobytes())
    return digest.hexdigest()


def _areas(p):
    a = np.asarray(p['labels'], np.int32)
    return {int(k): int(np.sum(a == int(k))) for k in p['regions']}


def _adjacency(p):
    """四邻接的共享像素边数；共享边越多，边界越长。"""
    a = np.asarray(p['labels'], np.int32)
    lengths = {}
    for u, v in ((a[:, :-1], a[:, 1:]), (a[:-1, :], a[1:, :])):
        mask = (u != v) & (u > 0) & (v > 0)
        for x, y in zip(u[mask].tolist(), v[mask].tolist()):
            pair = tuple(sorted((int(x), int(y))))
            lengths[pair] = lengths.get(pair, 0) + 1
    return dict(sorted(lengths.items()))


def _palette(palette):
    if not isinstance(palette, list):
        raise Invalid('色板格式无效')
    colors = []
    for c in palette:
        rgb(c)
        colors.append(c.lower())
    if not 2 <= len(colors) <= MAX_COLORS:
        raise Invalid(f'自动配色需要2–{MAX_COLORS}个色板颜色')
    if len(set(colors)) != len(colors):
        raise Invalid('自动配色需要互不重复的色板颜色')
    return colors


def _lab_distances(colors, sources):
    palette_lab = _lab([rgb(c) for c in colors])
    source_lab = _lab([rgb(c) for c in sources])
    return np.linalg.norm(palette_lab[None, :, :] - source_lab[:, None, :], axis=2)


def _fixed_choices(p, ids, colors):
    """锁定区域必须使用当前颜色；颜色不在色板时明确报错。"""
    fixed = {}
    for r in ids:
        region = p['regions'][str(r)]
        if not region['locked']:
            continue
        color = region['color'].lower()
        if color not in colors:
            raise Invalid(f'锁定区域{r}的颜色{color}不在当前色板中；请把该色加入色板或先解除锁定')
        fixed[r] = colors.index(color)
    for u, v in _adjacency(p):
        if u in fixed and v in fixed and fixed[u] == fixed[v]:
            raise Invalid(f'锁定区域{u}与{v}相邻且同为{colors[fixed[u]]}，无法满足相邻隔色；请调整锁色或色板')
    return fixed


def _solve(ids, colors, cost, adjacency, fixed, mode):
    """混合整数规划。x[r,c] 选择颜色；省印色模式另有 q[c] 标记该色是否被使用。"""
    rows = len(ids)
    count = len(colors)
    index = {r: i for i, r in enumerate(ids)}
    variables = rows * count
    q_offset = variables
    size = variables + (count if mode == 'inks' else 0)
    lower = np.zeros(size)
    upper = np.ones(size)
    for r, c in fixed.items():
        o = index[r] * count
        lower[o:o + count] = 0
        upper[o:o + count] = 0
        lower[o + c] = 1
        upper[o + c] = 1

    matrix, lows, highs = [], [], []
    for i in range(rows):
        row = np.zeros(size)
        row[i * count:(i + 1) * count] = 1
        matrix.append(row); lows.append(1); highs.append(1)
    for (u, v), _ in adjacency.items():
        for c in range(count):
            row = np.zeros(size)
            row[ids.index(u) * count + c] = 1
            row[ids.index(v) * count + c] = 1
            matrix.append(row); lows.append(-np.inf); highs.append(1)
    if mode == 'inks':
        for c in range(count):
            for i in range(rows):
                row = np.zeros(size)
                row[q_offset + c] = 1
                row[i * count + c] = -1
                matrix.append(row); lows.append(0); highs.append(np.inf)
    constraints = LinearConstraint(np.asarray(matrix), np.asarray(lows), np.asarray(highs))
    bounds = Bounds(lower, upper)
    tie = np.tile(np.arange(count, dtype=float) * 1e-7, rows)
    if mode == 'inks':
        c1 = np.zeros(size); c1[q_offset:] = 1
        first = milp(c1, integrality=np.ones(size), bounds=bounds, constraints=constraints, options={'time_limit': 20, 'mip_rel_gap': 0})
        if first.x is None:
            return None
        used = int(round(float(np.sum(first.x[q_offset:]))))
        matrix.append(np.concatenate([np.zeros(variables), np.ones(count)]))
        lows.append(used); highs.append(used)
        constraints = LinearConstraint(np.asarray(matrix), np.asarray(lows), np.asarray(highs))
        target = np.concatenate([cost.reshape(-1), np.zeros(count)]) + np.concatenate([tie, np.zeros(count)])
    else:
        used = None
        target = cost.reshape(-1) + tie
    result = milp(target, integrality=np.ones(size), bounds=bounds, constraints=constraints, options={'time_limit': 20, 'mip_rel_gap': 0})
    if result.x is None:
        if result.status == 2:
            return None
        raise Invalid('配色求解超时或未能完成，请减少区域或色板颜色后重试')
    x = np.rint(result.x[:variables]).astype(int).reshape(rows, count)
    if np.any(x.sum(axis=1) != 1):
        return None
    chosen = {r: colors[int(np.argmax(x[i]))] for i, r in enumerate(ids)}
    return chosen, int(np.sum(x.sum(axis=0) > 0))


def _check(p, ids, colors, chosen, adjacency, fixed):
    """采纳前的独立复核；任何一项不通过都不允许写入。"""
    violations = []
    for r in ids:
        if chosen.get(r) not in colors:
            violations.append(f'区域{r}取色不在色板内')
        elif p['regions'][str(r)]['locked'] and chosen[r] != p['regions'][str(r)]['color'].lower():
            violations.append(f'锁定区域{r}颜色被改变')
    if ids and any(r not in chosen for r in ids):
        violations.append('方案缺少区域')
    conflicts = []
    for (u, v), _ in adjacency.items():
        if chosen.get(u) == chosen.get(v):
            conflicts.append(dict(pair=[u, v], color=chosen.get(u)))
    if conflicts:
        violations.append('相邻区域同色')
    if fixed and any(chosen.get(r) != colors[c] for r, c in fixed.items()):
        violations.append('锁定区域未保持原色')
    return violations, conflicts


def plan(p, body):
    """生成只读方案。"""
    mode = body.get('mode', 'tone')
    if mode not in MODES:
        raise Invalid('请选择保留色调或省印色')
    colors = _palette(p['palette'])
    ids = sorted(int(k) for k in p['regions'])
    if len(ids) > MAX_REGIONS:
        raise Invalid(f'自动配色最多支持{MAX_REGIONS}个区域，当前{len(ids)}个；请先合并区域')
    if len(colors) != len(p['palette']):
        raise Invalid('色板含有重复颜色，自动配色需要互不重复的2–6色')
    areas = _areas(p)
    adjacency = _adjacency(p)
    fixed = _fixed_choices(p, ids, colors)
    sources = [p['regions'][str(r)]['source_color'].lower() for r in ids]
    distance = _lab_distances(colors, sources)
    # 以面积加权：大面积区域的偏色在总误差中更重，优先保持其原色。
    weight = np.array([areas[r] for r in ids], float)
    cost = distance * weight[:, None]
    solved = _solve(ids, colors, cost, adjacency, fixed, mode)
    if solved is None:
        raise Invalid('当前色板与锁色条件无法满足相邻隔色；请增加色板颜色、调整锁定区域或减少相邻约束')
    chosen, used = solved
    violations, conflicts = _check(p, ids, colors, chosen, adjacency, fixed)
    if violations:
        raise Invalid('方案未通过检查：' + '；'.join(dict.fromkeys(violations)))
    if mode == 'inks':
        used = len(set(chosen.values()))
    regions = []
    edge_checks = []
    for r in ids:
        source = p['regions'][str(r)]['source_color'].lower()
        regions.append(dict(id=r, name=p['regions'][str(r)].get('name', ''), area=areas[r], source_color=source,
                            color=chosen[r], deltas=round(float(_lab_distances([chosen[r]], [source])[0][0]), 2),
                            locked=bool(p['regions'][str(r)]['locked'])))
    for (u, v), length in adjacency.items():
        edge_checks.append(dict(pair=[u, v], edge_pixels=length, colors=[chosen[u], chosen[v]], ok=chosen[u] != chosen[v]))
    total_weight = float(np.sum(weight))
    weighted_delta = float(np.sum(distance[np.arange(len(ids)), [colors.index(chosen[r]) for r in ids]] * weight)) / total_weight
    large = max(ids, key=lambda r: (areas[r], -r))
    large_delta = round(float(_lab_distances([chosen[large]], [p['regions'][str(large)]['source_color'].lower()])[0][0]), 2)
    usage = {c: [r for r in ids if chosen[r] == c] for c in colors if any(chosen[r] == c for r in ids)}
    off_palette = sorted({chosen[r] for r in ids if chosen[r] not in colors})
    locked_ok = all(chosen[r] == p['regions'][str(r)]['color'].lower() for r in fixed)
    checks = [dict(label='相邻隔色', ok=not conflicts, detail=f'{len(adjacency)}组共享像素边均不同色'),
              dict(label='仅用当前色板', ok=not off_palette, detail=f'使用{used}种色板颜色，全部来自当前色板' if not off_palette else '发现色板外颜色：' + '、'.join(off_palette)),
              dict(label='锁色保持', ok=locked_ok, detail=f'{len(fixed)}个锁定区域保持原色' if locked_ok else '锁定区域被改动')]
    return dict(mode=mode, mode_label=MODE_LABELS[mode], palette=colors, colors={str(r): chosen[r] for r in ids},
                color_count=used, usage=usage, locked=[dict(id=r, color=chosen[r], kept=True) for r in ids if r in fixed],
                regions=regions, adjacency=dict(pairs=len(adjacency), passed=not conflicts, conflicts=conflicts, edges=edge_checks),
                mean_delta=round(weighted_delta, 2), large_region=dict(id=large, area=areas[large], delta=large_delta),
                checks=checks, fingerprint=_fingerprint(p))


def render_plan(p, result):
    """预览 PNG：只在内存中套用方案颜色，不修改项目。"""
    if not isinstance(result, dict) or result.get('fingerprint') != _fingerprint(p):
        raise Conflict(STALE)
    chosen = result.get('colors')
    if not isinstance(chosen, dict):
        raise Invalid('方案缺少配色')
    for key, region in p['regions'].items():
        if key in chosen:
            rgb(chosen[key])
    image = render(p)
    a = np.asarray(p['labels'])
    for key, color in chosen.items():
        if str(key) in p['regions']:
            image[a == int(key)] = (*rgb(color), 255)
    return png(image)


def apply(p, result):
    """采纳方案：复核指纹与全部约束后作为一次编辑写入。"""
    if not isinstance(result, dict) or result.get('fingerprint') != _fingerprint(p):
        raise Conflict(STALE)
    if result.get('conflicts') or any(not c.get('ok', False) for c in result.get('checks', [])):
        raise Invalid('方案未通过检查，不能采纳')
    colors = _palette(p['palette'])
    ids = sorted(int(k) for k in p['regions'])
    if len(ids) > MAX_REGIONS:
        raise Invalid(f'自动配色最多支持{MAX_REGIONS}个区域')
    chosen = result.get('colors')
    if not isinstance(chosen, dict) or set(chosen) != {str(r) for r in ids}:
        raise Invalid('方案区域与当前区域不一致，请重新生成')
    fixed = _fixed_choices(p, ids, colors)
    normalized = {int(k): str(v).lower() for k, v in chosen.items()}
    violations, _ = _check(p, ids, colors, normalized, _adjacency(p), fixed)
    if violations:
        raise Invalid('方案未通过检查：' + '；'.join(dict.fromkeys(violations)))
    checkpoint(p)
    for r in ids:
        p['regions'][str(r)]['color'] = normalized[r]
    return p
