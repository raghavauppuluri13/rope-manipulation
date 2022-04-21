import grp
from rope_env import rope_env,FRONT_VIEW,EYE_IN_HAND_VIEW,TASK_VIEW
from collections import deque
from observer import Observer
from planner import ArmPlanner, GripperPlanner, Planner
from controllers import ArmPositionController, GripperController

env = rope_env()
env.physics.contexts.mujoco.free()
#viewer.launch(env)

timestep = env.reset()
obs = timestep.observation['rope/position']
hover = obs[:,9,:] + [0,0,0.1]
pick = obs[:,9,:]

place = pick + [-0.3,-0.3,0] 
home = place + [0,0,0.6] 

observer = Observer(env,obs_camera=TASK_VIEW,show=False)

arm_controller = ArmPositionController(env,'config/panda.yaml')
arm_planner = ArmPlanner(env,arm_controller,observer,interpolator_step=0.02) 

grip_controller = GripperController(env,'config/panda_hand.yaml')
grip_planner = GripperPlanner(env,grip_controller,observer,interpolator_step=0.05) 

setpoints = deque([
    (arm_planner,home),
    (grip_planner,grip_controller.open),
    (arm_planner,place),
    (grip_planner,grip_controller.close),
    (arm_planner,pick),
    (arm_planner,hover),
    (grip_planner,grip_controller.open)
])

while setpoints or not arm_planner.done or not grip_planner.done:
    if arm_planner.done and grip_planner.done:
        planner,setp = setpoints.pop()
        planner.set_goal(setp)
    arm_action = arm_planner.get_action()
    grip_action = grip_planner.get_action()
    action = arm_action + grip_action
    timestep = env.step(action)
    observer.step(timestep)

observer.save("exploration.gif")