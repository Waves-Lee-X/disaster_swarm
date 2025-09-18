import numpy as np

class BoidsPlanner:
    def __init__(self, drones, max_speed=3.0, sep_w=1.5, align_w=0.8, coh_w=0.6, sep_radius=5.0, neigh_radius=12.0):
        self.drones = drones
        self.max_speed = float(max_speed)
        self.sep_w, self.align_w, self.coh_w = map(float, (sep_w, align_w, coh_w))
        self.sep_radius = float(sep_radius)
        self.neigh_radius = float(neigh_radius)

    def compute(self, positions, velocities):
        cmds = {}
        for i, d in enumerate(self.drones):
            p_i = positions[d]
            v_i = velocities[d]
            sep = np.zeros(3, dtype=float)
            align = np.zeros(3, dtype=float)
            coh_sum = np.zeros(3, dtype=float)
            n_align = 0
            n_coh = 0
            for j, dj in enumerate(self.drones):
                if dj == d: continue
                dp = positions[dj] - p_i
                dist = np.linalg.norm(dp)
                if dist < 1e-6: continue
                if dist < self.sep_radius:
                    sep -= dp / (dist**2 + 1e-6)
                if dist < self.neigh_radius:
                    align += velocities[dj]
                    coh_sum += positions[dj]
                    n_align += 1; n_coh += 1
            if n_align > 0:
                align = align / n_align - v_i
            if n_coh > 0:
                coh = (coh_sum / n_coh) - p_i
            else:
                coh = np.zeros(3, dtype=float)

            cmd = self.sep_w * sep + self.align_w * align + self.coh_w * coh
            nrm = np.linalg.norm(cmd)
            if nrm > self.max_speed: cmd = cmd / nrm * self.max_speed
            cmds[d] = cmd
        return cmds
