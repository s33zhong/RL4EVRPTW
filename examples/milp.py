from os import makedirs, path
import numpy as np
import torch
from torch import tensor
import gurobipy as gp
from gurobipy import GRB


def check_within_ev_limit(actions, num_ev):
    return (actions == 0).sum().item() <= num_ev

def milp_solver(td, env, timelimit=60):
    """Variables for MILP Solver

    Vertices and Arcs
    V: Customer nodes
    F' (Fp): Augmented station nodes (dummy vertices for multiple visits)
    V' (Vp): All vertices (customers + stations + depot at 0 and last+1)
    A[i, j]: Undirected arcs between pairs of vertices
    D[i, j]: Distance of arc (Euclidean distance)
    T[i, j]: Travel time for each arc

    Time
    H: Problem horizon (maximum timeframe)
    e: Earliest time window for each vertex
    l: Latest time window for each vertex
    s: Service times (set to zero for simplification)
    g: Recharge rate (energy recharged per unit time at stations)
    M: Upper bound for time constraints, M = l_0 + gQ

    Demand
    C: Maximum cargo capacity of the vehicle
    d: Customer demands

    Fuel
    Q: Maximum battery energy
    h: Energy consumption rate per unit distance

    Decision Variables
    a[i, j]: Binary variable, 1 if arc (i, j) is traveled, 0 otherwise
    tau[i]: Continuous variable for arrival time at vertex i
    c[i]: Continuous variable for remaining cargo after visiting vertex i
    b[i]: Continuous variable for remaining energy after visiting vertex i
    """
    td = {key: tensor.cpu().numpy() for key, tensor in td.clone().items()}
    
    num_stations = td['stations'].shape[0]
    num_customers = td['demand'].shape[0]
    num_freq = num_customers//num_stations

    if td['locs'].shape[0] != num_stations + num_customers + 1:
        raise ValueError(
            f"Invalid number of locations. Expected {num_stations + num_customers + 1}, "
            f"but got {td['locs'].shape[0]}."
        )
        
    V = [v_i for v_i in range(1, num_customers + 1)]
    Fp = [v_i for v_i in range(num_customers + 1, num_customers + 1 + num_stations * num_freq)]
    Vp = V+Fp
    Vp = [0] + Vp + [max(Vp) + 1]
    A = [(i, j) for i in Vp for j in Vp if i != j]
    
    def add_dummy_stations(arr, num_stations, num_freq):
        arr = arr.copy()
        arr = np.vstack([arr[:-num_stations], np.repeat(arr[-num_stations:], num_freq, axis=0), arr[0]])
        return arr
        
    locs = add_dummy_stations(td['locs'], num_stations, num_freq)
    time_windows = add_dummy_stations(td['time_windows'], num_stations, num_freq)
    D = np.linalg.norm(locs[:, np.newaxis] - locs, axis=2)
    T = D / env.generator.vehicle_speed

    H = env.generator.horizon
    e = time_windows[:, 0].tolist()
    l = time_windows[:, 1].tolist()
    s = [0 for _ in Vp]
    g = env.generator.inverse_recharge_rate
    C = td['vehicle_capacity'][0]
    d = {i: td['demand'][i-1] if i in V else 0 for i in Vp}
    Q = env.generator.max_fuel * env.generator.fuel_consumption_rate
    h = env.generator.fuel_consumption_rate
    M = l[0] + g * Q
    
    model = gp.Model("EVRPTW")

    a = model.addVars(A, vtype=GRB.BINARY, name='a')
    tau = model.addVars(Vp, vtype=GRB.CONTINUOUS, lb=0, ub=H, name='tau')
    c = model.addVars(Vp, vtype=GRB.CONTINUOUS, lb=0, ub=C, name='c')
    b = model.addVars(Vp, vtype=GRB.CONTINUOUS, lb=0, ub=Q, name='b')

    obj = gp.quicksum(D[(i,j)] * a[i, j] for i in Vp[:-1] for j in Vp[1:] if i!=j)

    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constaints
    model.addConstrs(gp.quicksum(a[i, j] for j in Vp[1:] if i != j) == 1 for i in V)
    model.addConstrs(gp.quicksum(a[i, j] for j in Vp[1:] if i != j) <= 1 for i in Fp)
    model.addConstrs(gp.quicksum(a[j, i] for i in Vp[1:] if i != j) -
                    gp.quicksum(a[i, j] for i in Vp[:-1] if i != j) == 0
                    for j in Vp[1:-1])
    model.addConstrs(tau[i]+(T[(i,j)]+s[i])*a[i,j]-l[0]*(1-a[i,j]) <= tau[j] for i in [0]+V for j in Vp[1:] if i!=j)
    model.addConstrs(tau[i]+T[(i,j)]*a[i,j] + g*(Q-b[i]) - M*(1-a[i,j])<=tau[j] for i in Fp for j in Vp[1:] if i!=j)
    model.addConstrs(c[j]<=c[i]-d[i]*a[i,j]+C*(1-a[i,j]) for i in Vp[:-1] for j in Vp[1:] if i!=j)
    model.addConstrs(b[j]<=b[i]-(h*D[(i, j)])*a[i,j] + Q*(1-a[i,j]) for j in Vp[1:] for i in V if i!=j)
    model.addConstrs(b[j]<=Q-(h*D[(i, j)])*a[i,j] for j in Vp[1:] for i in [0]+Fp if i!=j)
    model.addConstrs((e[j] <= tau[j] for j in Vp), name="10a")
    model.addConstrs((tau[j] <= l[j] for j in Vp), name="10b")
    model.addConstr(tau[0] == 0)
    model.addConstr(c[0] == C)
    model.addConstr(b[0] == Q)
    model.addConstr(gp.quicksum(a[0, j] for j in Vp[1:-1]) <= env.generator.vehicle_limit)
    
    model.setParam('TimeLimit', timelimit)
    model.optimize()
    
    if model.SolCount > 0:
        X = np.zeros((len(Vp)+1, len(Vp)+1))
        for (i,j), val in a.items():
            X[i, j] = round(val.x)
            
        actions = []
        for n in np.where(X[0] == 1)[0]:
            actions.extend([0, n])
            index = np.where(X[n] == 1)[0] 
            while index.size > 0:
                if index.size > 1:
                    raise ValueError("Multiple stations visited at the same time, cannot determine sequence")
                n = index[0]
                actions.append(n)
                index = np.where(X[n] == 1)[0]
        
        actions = np.array(actions)
        actions = actions[actions!=max(actions)]
        actions = np.where(
            actions >= num_customers,
            num_customers + 1 + (actions - num_customers - 1) // num_freq,
            actions
        )

        result = {'model': model,
                'X': X.astype(int),
                'actions': tensor(actions),
                'is_feasible': check_within_ev_limit(actions, env.generator.vehicle_limit)}
    
    else:
        result = {'model': model,
                'X': None,
                'actions': None,
                'is_feasible': False}
    
    return result

def batch_milp(td, env, num_loc, num_station, num_ev, save=True, timelimit=60):
    td = td.clone()
    runtime = 0
    actions = {}
    feasible_instances = []
    
    for i in range(td.shape[0]):
        print(f"\nProcessing instance {i+1}/{td.shape[0]} of C{num_loc}-S{num_station}-EV{num_ev}\n")
        
        result = milp_solver(td[i], env, timelimit=timelimit)
        
        action = result['actions']
        actions[i] = action
        
        if result['is_feasible']:
            runtime +=  result['model'].Runtime
            feasible_instances.append(i)
    
    num_feasibles=len(feasible_instances)
    if num_feasibles > 0:
        valid_indices, valid_actions = zip(*[(i, v) for i, v in actions.items() if i in feasible_instances])
        td_valid = td.clone()[list(valid_indices)]
        max_len = max(v.size(0) for v in valid_actions)
        actions_tensor = torch.stack([
            torch.cat([v, torch.zeros(max_len - v.size(0), dtype=torch.long)]) if v.size(0) < max_len else v
            for v in valid_actions
        ])
        
        rewards = env.get_reward(td_valid, actions_tensor.cuda())
        mean_reward = rewards.mean().item()
        avg_runtime = runtime/num_feasibles
    else:
        mean_reward = 0
        avg_runtime = 0
        print("No feasible instances")
    
    if save:
        makedirs("save_milp", exist_ok=True)
        filename = f"actions_c{num_loc:03d}_s{num_station}_v{num_ev}_b{td.shape[0]}.pt"
        torch.save(actions, path.join("save_milp", filename))
        
    return num_loc, num_station, num_ev, num_feasibles, -mean_reward, avg_runtime

def check_actions_with_mask(td, env, actions, display_errors=False):
    check_count=0
    def check_visited(td, actions, batch_index):
        td_test = td[batch_index: batch_index+1].clone()
        for id, action in enumerate(actions[1:], start=1):
            # skip multiple visit to same nodes, likely stations
            if actions[id]==actions[id-1]:
                continue
            # skip check when visit depot or station
            if action==0 or action>td['demand'].shape[-1] or td_test['action_mask'][0][action].item():
                td_test['action'] = torch.tensor([action.item()])
                td_test = env.step(td_test)['next'] 
            else:
                if display_errors:
                    c_time = td_test['current_time'].item()
                    time_to_go = (td_test['distances'][0][action]/td_test['vehicle_speed']).item()
                    print(f"Current Time: {c_time:.9f} ", 
                        f"Need: {time_to_go:.9f}",
                        f"Reach: {c_time+time_to_go:.9f}")
                    print("Time Windows", td_test['time_windows'][0][action].tolist())
                    print("Current Fuel", f"{td_test['current_fuel'].item()}")
                    print("Current Capacity", 
                        f"{td_test['vehicle_capacity'].item()-td_test['used_capacity'].item()}")
                    print("Demand needed:", td_test['demand'][0][action-1].item())
                break
        return td_test['finished'], id

    invalid_results = []
    for k, act in actions.items():
        if act.tolist().count(0)<=env.generator.vehicle_limit:
            check_count+=1
            success, fail_id = check_visited(td, act, k)
            if not success:
                invalid_results.append(k)
                print(f"Fail at instance {k} at node {act[fail_id]}")
    print(f"Checked {check_count} results")
    if check_count>0 and len(invalid_results)==0:
        print("All results are valid.")
    elif check_count==0:
        print("No feasible results to check.")
    else:
        print("Invalid instances no.:", " ".join(map(str, invalid_results)))

