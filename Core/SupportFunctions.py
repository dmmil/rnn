from typing import List, Dict

def init_route(route_type: int, L: int, M: int) -> List[Dict]:
    route = []
    if route_type == 0:
        right_direction = False
        for il in range(L):
            right_direction = not right_direction
            if right_direction:
                for im in range(M):
                    route.append({'Lr': 0, 'L': il, 'M': im})
                    route.append({'Lr': 1, 'L': il, 'M': im})
            else:
                for im in reversed(range(M)):
                    route.append({'Lr': 0, 'L': il, 'M': im})
                    route.append({'Lr': 1, 'L': il, 'M': im})
    else:
        raise 'selected route type not supported in this version'
    return route

