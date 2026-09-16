"""Label-free original-link protection supported by three-frame motion."""


def protected_links(nodes,positions,probabilities,minimum_probability,max_distance,max_velocity_change):
    import math
    from collections import Counter,defaultdict
    if (not 0<=minimum_probability<=1 or not math.isfinite(max_distance) or max_distance<=0
            or not math.isfinite(max_velocity_change) or max_velocity_change<0):
        raise ValueError('Invalid protection thresholds')
    eligible=[]
    for (a,b),value in probabilities.items():
        if a not in nodes or b not in nodes or int(nodes[b]['t'])!=int(nodes[a]['t'])+1:
            continue
        try:p=float(value)
        except (TypeError,ValueError):continue
        if not math.isfinite(p) or not minimum_probability<=p<=1:continue
        distance=math.dist(positions[a],positions[b])
        if not math.isfinite(distance) or distance>max_distance:continue
        eligible.append((a,b,distance,p))
    sources=Counter(a for a,b,d,p in eligible);targets=Counter(b for a,b,d,p in eligible)
    unique={(a,b):(d,p) for a,b,d,p in eligible if sources[a]==1 and targets[b]==1}
    successor={a:b for a,b in unique}
    supported=set()
    for a,b in sorted(unique):
        c=successor.get(b)
        if c is None:continue
        incoming=tuple(float(y)-float(x) for x,y in zip(positions[a],positions[b]))
        outgoing=tuple(float(y)-float(x) for x,y in zip(positions[b],positions[c]))
        if math.dist(incoming,outgoing)<=max_velocity_change:
            supported.add((a,b));supported.add((b,c))
    by_time=defaultdict(list)
    for a,b in sorted(supported):
        d,p=unique[a,b]
        by_time[int(nodes[a]['t'])].append((a,b,d,p))
    return dict(by_time)
