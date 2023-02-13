
import habitat_sim

GOAL_CATEGORIES = [
    "chair",
    "bed",
    "plant",
    "toilet",
    "tv_monitor",
    "sofa",
]

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = (
    "data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
)
backend_cfg.scene_dataset_config_file = (
    "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
)

rgb_cfg = habitat_sim.CameraSensorSpec()
rgb_cfg.uuid = "rgb"
rgb_cfg.sensor_type = habitat_sim.SensorType.COLOR


sem_cfg = habitat_sim.CameraSensorSpec()
sem_cfg.uuid = "semantic"
sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb_cfg, sem_cfg]

sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(sim_cfg)

total_objects = len(sim.semantic_scene.objects)
print("Total objects - {}".format(total_objects))

for obj in sim.semantic_scene.objects:
    if obj.category.name("") in GOAL_CATEGORIES:
        print(obj.category.name(""), obj.obb.center, obj.aabb.center, obj.obb.sizes)
   
