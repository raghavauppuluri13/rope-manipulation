import os
from dm_control import mjcf


class BaseEnv:
    def __init__(self):
        self.obj = mjcf.from_path(os.path.abspath("models/arenas/checkers.xml"))
        self.physics = mjcf.Physics.from_mjcf_model(self.obj)


class RopeEnv(BaseEnv):
    def __init__(self):
        super().__init__()

        self.obj.worldbody.add(
            "camera", name="overhead", pos="0 -0.4 1", quat="1 0 0 0"
        )

        panda = mjcf.from_path(os.path.abspath("models/panda.xml"))
        panda.worldbody.body["panda_link0"].pos = [0, 0, 0]
        panda.worldbody.body["panda_link0"].quat = [0.7071068, 0, 0, -0.7071068]
        self.obj.attach(panda)

        rope = mjcf.from_path(os.path.abspath("models/rope/rope.xml"))
        rope.worldbody.body["CB0"].pos = [0, -0.4, 0]
        self.obj.attach(rope)
        # peg = mjcf.from_path(os.path.abspath("models/peg/peg.xml"))
        # peg.worldbody.body["peg_body"].pos = [0, -0.5, 0]
        # self.obj.attach(peg)

        self.physics = mjcf.Physics.from_mjcf_model(self.obj)
