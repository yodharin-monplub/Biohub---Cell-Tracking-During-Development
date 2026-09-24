"""Label-free, one-to-one original-link anchors for motion assignment."""


def protected_links(nodes,positions,probabilities,minimum_probability,max_distance):
    import math
    from collections import Counter,defaultdict
    if not 0 <= minimum_probability <= 1 or not math.isfinite(max_distance) or max_distance<=0:
        raise ValueError('Invalid protection thresholds')
    eligible=[]
    for (a,b),value in probabilities.items():
        if a not in nodes or b not in nodes or int(nodes[b]['t'])!=int(nodes[a]['t'])+1:
            continue
        try:
            p=float(value)
        except (TypeError,ValueError):
            continue
        if not math.isfinite(p) or not minimum_probability<=p<=1:
            continue
        distance=math.dist(positions[a],positions[b])
        if not math.isfinite(distance) or distance>max_distance:
            continue
        eligible.append((a,b,distance,p))
    sources=Counter(a for a,b,d,p in eligible);targets=Counter(b for a,b,d,p in eligible)
    by_time=defaultdict(list)
    for a,b,d,p in sorted(eligible):
        if sources[a]==1 and targets[b]==1:
            by_time[int(nodes[a]['t'])].append((a,b,d,p))
    return dict(by_time)
