"""Mixed integer program solved at each time step for model predictive control"""

from typing import Dict, Tuple, List
import gurobipy
from gurobipy import GRB, Var, Constr, LinExpr, quicksum


Weight = int
Time = int
Arc = Tuple[int, int, Weight, Time]
TimeWindow = Tuple[int, int]
ServiceTime = int


class ResourceConstrainedFlowModel:
    _model: gurobipy.Model
    _arc_decisions: Dict[Tuple[int, int], Var]

    # input
    _arcs: List[Arc]
    _time_windows: List[TimeWindow]
    _service_times: List[ServiceTime]
    _source: int
    _sink: int

    _num_nodes: int
    _num_arcs: int

    # helpers
    _begin_of_service: List[Var]
    _departure_time: List[Var]
    _flow_conservation: Dict[Tuple[int, int], Constr]
    _objective: LinExpr

    def __init__(self,
                 arcs: List[Arc],
                 tws: List[TimeWindow],
                 service_times: List[ServiceTime],
                 source_id=0,
                 sink_id=None,
                 env: gurobipy.Env = gurobipy.Env()):

        if sink_id is None:
            sink_id = len(tws) - 1

        m = self._model = gurobipy.Model(env=env)

        self._arcs = arcs
        self._time_windows = tws
        self._service_times = service_times
        self._source = source_id
        self._sink = sink_id
        num_nodes = self._num_nodes = len(tws)
        num_arcs = self._num_arcs = len(arcs)

        self._arc_decisions = dict()
        self._begin_of_service = list()
        self._departure_time = list()
        self._flow_conservation = dict()

        big_m = max(map(lambda it: it[1], tws))

        for (u, v, weight, _) in arcs:
            self._arc_decisions[(u, v)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.INTEGER, name=f"x_{u},{v}", obj=weight)

        for u, (start, end) in enumerate(tws):
            self._begin_of_service.append(m.addVar(lb=start, ub=end, vtype=GRB.INTEGER, name=f"tau_{u}"))
            self._departure_time.append(m.addVar(lb=0.0, ub=big_m, vtype=GRB.INTEGER, name=f"d_{u}"))

        # flow conservation

        # create helper structures
        incoming = [[] for _ in range(num_nodes)]
        outgoing = [[] for _ in range(num_nodes)]
        for arc in arcs:
            origin, destination, _, _ = arc
            outgoing[origin].append(arc)
            incoming[destination].append(arc)

        # create constraints
        for v in range(num_nodes):
            m.addConstr(self._begin_of_service[v] + service_times[v] <= self._departure_time[v], name=f"c_{v}")
            
            if v == source_id:
                pass
            elif v == sink_id:
                for (u, _, _, time) in incoming[v]:
                    x = self._arc_decisions[(u, v)]
                    m.addConstr(self._departure_time[u] + time <= tws[v][1] + (1 - x) * big_m, name=f"t1_{u},{v}")
                    m.addConstr(self._departure_time[u] + (1 - x) * big_m >= tws[v][0], name=f"t2_{u},{v}")
            else:
                sum_incoming = quicksum(map(lambda arc: self._arc_decisions[arc[0], v], incoming[v]))
                m.addConstr(sum_incoming <= 1, name=f"maxInFlow_{v}")

                sum_outgoing = quicksum(map(lambda arc: self._arc_decisions[v, arc[1]], outgoing[v]))
                m.addConstr(sum_outgoing <= 1, name=f"maxOutFlow_{v}")

                m.addConstr(sum_incoming == sum_outgoing, f"FlowConservation_{v}")

                for (u, _, _, time) in incoming[v]:
                    x = self._arc_decisions[(u, v)]
                    m.addConstr(self._departure_time[u] + time <= self._begin_of_service[v] + (1 - x) * big_m, name=f"t1_{u},{v}")
                    m.addConstr(self._departure_time[u] + (1 - x) * big_m >= tws[v][0], name=f"t2_{u},{v}")

        m.setObjective(
            quicksum(map(lambda arc: self._arc_decisions[(arc[0], arc[1])] * arc[2], arcs)),
            GRB.MINIMIZE)

    def optimize(self) -> Tuple[int, int]:
        self._model.setParam("TimeLimit", 50.0)
        self._model.optimize()
        
        selected_arcs = list()
        for u, v in self._arc_decisions:
            if abs(self._arc_decisions[(u, v)].x) == 1:
                selected_arcs.append((u, v))

        return self._model.getObjective().getValue(), self._model.ObjBound, selected_arcs
