import pybullet_data
import numpy as np

import utils
import manipulators


class GenericEnv:
    def __init__(self, gui=0, seed=None):
        self._p = utils.connect(gui)
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self._p.resetSimulation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.807)
        self._p.loadURDF("plane.urdf")

        self.env_dict = utils.create_tabletop(self._p)
        self.agent = manipulators.Manipulator(p=self._p, path="ur10e/ur10e.urdf", position=[0., 0., 0.4], ik_idx=6)
        base_constraint = self._p.createConstraint(parentBodyUniqueId=self.env_dict["base"], parentLinkIndex=0,
                                                   childBodyUniqueId=self.agent.id, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0),
                                                   childFramePosition=(0.0, 0.0, -0.2),
                                                   childFrameOrientation=(0, 0, 0, 1))
        self._p.changeConstraint(base_constraint, maxForce=10000)

        self.tool_id = utils.create_object(self._p, self._p.GEOM_CYLINDER, size=[0.05, 0.05], position=[0, 0, 0], mass=1.0, color=[0.2, 0.2, 0.2, 1.0])
        tool_constraint = self._p.createConstraint(parentBodyUniqueId=self.agent.id, parentLinkIndex=6,
                                                   childBodyUniqueId=self.tool_id, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0), childFramePosition=(0, 0, 0.03),
                                                   childFrameOrientation=self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]))
        self._p.changeConstraint(tool_constraint, maxForce=10000)

    def init_agent_pose(self, t=None, sleep=False, traj=False):
        angles = [-0.294, -1.650, 2.141, -2.062, -1.572, 1.277]
        self.agent.set_joint_position(angles, t=t, sleep=sleep, traj=traj)

    def _step(self, count=1):
        for _ in range(count):
            self._p.stepSimulation()


class PushEnv(GenericEnv):
    def __init__(self, gui=0, seed=None):
        super(PushEnv, self).__init__(gui=gui, seed=seed)

    def reset(self, seed=None):
        super(PushEnv, self).reset(seed=seed)

        self.obj_dict = {}
        self.init_agent_pose(t=1)
        self.init_objects()
        self._step(40)

    def reset_objects(self):
        for key in self.obj_dict:
            obj_id = self.obj_dict[key]
            self._p.removeBody(obj_id)
        self.obj_dict = {}
        self.init_objects()
        self._step(240)

    def init_objects(self):
        obj_type = np.random.choice([self._p.GEOM_BOX, self._p.GEOM_SPHERE, self._p.GEOM_CYLINDER], p=[0.6, 0.1, 0.3])
        position = [0.8, 0.0, 0.6]
        rotation = [0, 0, 0]
        if obj_type == self._p.GEOM_CYLINDER:
            r = np.random.uniform(0.05, 0.1)
            h = np.random.uniform(0.05, 0.1)
            size = [r, h]
            if np.random.rand() < 0.99:
                rotation = [np.pi/2, 0, 0]
        else:
            r = np.random.uniform(0.025, 0.05)
            size = [r, r, r]

        self.obj_dict[0] = utils.create_object(p=self._p, obj_type=obj_type, size=size, position=position,
                                               rotation=rotation, color="random", mass=1.0)

    def state(self):
        rgb, depth, seg = utils.get_image(p=self._p, eye_position=[1.5, 0.0, 1.5], target_position=[0.9, 0., 0.4],
                                          up_vector=[0, 0, 1], height=256, width=256)
        return rgb[:, :, :3], depth, seg

    def step(self, action, sleep=False):
        traj_time = 1

        if action == 0:
            self.agent.set_cartesian_position([0.8, -0.15, 0.75], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.8, -0.15, 0.46], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.8, 0.15, 0.46], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=traj_time, sleep=sleep)
            self.agent.set_cartesian_position([0.8, 0.15, 0.75], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep, traj=True)
            self.init_agent_pose(sleep=sleep)
        elif action == 1:
            self.agent.set_cartesian_position([0.95, 0.0, 0.75], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.95, 0.0, 0.46], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.65, 0.0, 0.46], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=traj_time, sleep=sleep)
            self.agent.set_cartesian_position([0.65, 0.0, 0.75], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep, traj=True)
            self.init_agent_pose(sleep=sleep)
        elif action == 2:
            self.agent.set_cartesian_position([0.8, 0.15, 0.75], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.8, 0.15, 0.46], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.8, -0.15, 0.46], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=traj_time, sleep=sleep)
            self.agent.set_cartesian_position([0.8, -0.15, 0.75], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep, traj=True)
            self.init_agent_pose(sleep=sleep)
        elif action == 3:
            self.agent.set_cartesian_position([0.65, 0.0, 0.75], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.65, 0.0, 0.46], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.95, 0.0, 0.46], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=traj_time, sleep=sleep)
            self.agent.set_cartesian_position([0.95, 0.0, 0.75], self._p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=0.5, sleep=sleep, traj=True)
            self.init_agent_pose(sleep=sleep)
