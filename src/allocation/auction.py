import numpy as np

def compute_bid(pos, task, energy=1.0, capability=1.0, w_dist=1.0, w_energy=0.3, w_cap=0.2):
    # 计算车辆对任务的出价分数，出价越高表示车辆越适合执行该任务
    # pos: 车辆位置(numpy数组，包含x, y, z)
    # task: 任务字典,包含x, y, z等信息
    # energy: 车辆剩余能量(0到1.默认为1)
    # capability：车辆能力分数
    # w_dist w_energy w_cap 距离 能量 能力权重
    dist = np.linalg.norm(pos[:2] - np.array([task['x'], task['y']])) # 计算车辆与任务的二维欧几里得距离
    bid = -w_dist * dist + w_energy * energy + w_cap * capability # 出价公式: 距离惩罚，能量和能力奖励
    return bid # 返回出价分数

def auction_task(client, tasks, vehicles, energy_map=None, capability_map=None):
    # 基于拍卖机制的任务分配函数，将任务分配给最合适的车辆
    # client: AirSim客户端实例，用于获取车辆状态和发送移动指令
    # tasks: 任务列表，每个任务是字典，包含id, x, y, z, type
    # vehicles: 车辆名称列表
    # energy_map: 字典，映射车辆到剩余能量 (0到1)，若未提供则默认1.0
    # capability_map: 字典，映射车辆到能力分数，若未提供则默认1.0

    #初始化能量和能力字典，若未提供则设为默认值1.0
    if energy_map is None:
        energy_map = {v: 1.0 for v in vehicles}
    if capability_map is None:
        capability_map = {v: 1.0 for v in vehicles}

    assignments = {} # 初始化分配结果字典，存储任务ID到车辆名称的映射
    pos_map = {} # 初始化位置字典，存储每辆车的当前位置
    # 一次性获取所有车辆的位置，避免重复查询AirSim
    for v in vehicles:
        s = client.getMultirotorState(vehicle_name=v) # 获取车辆状态
        p = s.kinematics_estimated.position # 获取位置信息
        pos_map[v] = np.array([p.x_val, p.y_val, p.z_val]) # 存储为numpy数组

    # 遍历每个任务，执行拍卖分配
    for task in tasks:
        best_bid = -1e9
        best_v = None
        # 遍历计算对当前任务的出价
        for v in vehicles:
            bid = compute_bid(pos_map[v], task, energy=energy_map.get(v,1.0), capability=capability_map.get(v,1.0))
            if bid > best_bid:
                best_bid = bid
                best_v = v

        # 如果找到最佳车辆，记录分配并发送移动指令
        if best_v is not None:
            assignments[task['id']] = best_v # 记录任务分配
            try:
                # 异步命令车辆移动到任务位置，速度为3m/s,z坐标默认为-5
                client.moveToPositionAsync(task['x'], task['y'], task.get('z', -5), 3, vehicle_name=best_v)
            except Exception as e:
                print(f"[auction] move failed: {e}") # 捕获并打印移动指令错误
    return assignments # 返回任务分配结果

