"""Render factorized, static multi-object evidence banks with Blender.

Run with Blender --background --python render_multistep.py -- --output DIR --seed N.
The JSON is evaluator-private. Public observations contain photographs and neutral
position addresses, never material seeds, permutations or geometric object IDs.
CPU is the default because GPU accumulation can differ by one output gray level.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np

SLOTS = ("left", "center", "right")
CENTERS = (-1.65, 0.0, 1.65)
FRONT_Y, REAR_Y = -0.456, 0.456
BOTTOM, TOP = 0.23, 1.33


def material(name, color):
    """Deterministic matte appearance: no stochastic reflection or light sampling."""
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1)
    value.use_nodes = True
    nodes = value.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    shade = nodes.new("ShaderNodeMath")
    shade.operation = "MULTIPLY_ADD"
    shade.inputs[1].default_value, shade.inputs[2].default_value = .35, .65
    multiply = nodes.new("ShaderNodeMixRGB")
    multiply.name, multiply.blend_type = "surface-color", "MULTIPLY"
    multiply.inputs[0].default_value = 1
    multiply.inputs[1].default_value = (*color, 1)
    links = value.node_tree.links
    links.new(geometry.outputs["Normal"], separate.inputs[0])
    links.new(separate.outputs["Z"], shade.inputs[0])
    links.new(shade.outputs[0], multiply.inputs[2])
    links.new(multiply.outputs[0], emission.inputs[0])
    links.new(emission.outputs[0], output.inputs[0])
    return value


def wood_material(seed):
    """Identical palette, identity carried by irregular grain and knot placement."""
    rng = np.random.default_rng(seed)
    size = 512
    yy, xx = np.mgrid[0:1:complex(size), 0:1:complex(size)]
    warp = xx + 0.011 * np.sin(yy * 24) + 0.007 * np.sin(yy * 61)
    knots = []
    for _ in range(3):
        cx, cy = rng.uniform(0.16, 0.84, 2)
        rx, ry = rng.uniform(0.035, 0.07), rng.uniform(0.065, 0.12)
        distance = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        warp += 0.024 * np.exp(-distance / 4) * np.sin(distance * 2)
        knots.append(np.exp(-distance * 1.5))
    grain = 0.045 * np.sin(warp * 260) + 0.018 * np.sin(warp * 680)
    tone = 0.80 + grain - 0.40 * np.maximum.reduce(knots)
    tone += rng.normal(0, 0.006, tone.shape)
    palette = np.array([0.49, 0.30, 0.15])
    pixels = np.ones((size, size, 4), dtype=np.float32)
    pixels[:, :, :3] = np.clip(tone[:, :, None] * palette, 0, 1)
    image = bpy.data.images.new(f"grain-{seed}", width=size, height=size)
    image.pixels.foreach_set(pixels.ravel())
    image.update()
    value = material(f"front-material-{seed}", tuple(palette))
    node = value.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    node.interpolation = "Linear"
    value.node_tree.links.new(node.outputs["Color"],
                             value.node_tree.nodes["surface-color"].inputs[1])
    return value


def cube(name, location, dimensions, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    obj.data.materials.append(mat)
    return obj


def face(name, x, y, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(x - .52, y, BOTTOM), (x + .52, y, BOTTOM),
                     (x + .52, y, TOP), (x - .52, y, TOP)], [], [(0, 1, 2, 3)])
    mesh.uv_layers.new()
    for loop, uv in zip(mesh.uv_layers.active.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        loop.uv = uv
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def scratch(name, x, coordinates, mat):
    """Flat exposed-wood damage with a darker edge, without a raised tube."""
    edge = material(name + "-edge", (.16, .09, .04))
    edge.pass_index = 3
    vertices, polygons, materials = [], [], []
    for width, offset, index in [(.006, .0004, 0), (.005, .0008, 1)]:
        for left, right in zip(coordinates, coordinates[1:]):
            dx, dz = right[0] - left[0], right[1] - left[1]
            nx, nz = -dz * width / math.hypot(dx, dz), dx * width / math.hypot(dx, dz)
            first = len(vertices)
            vertices.extend([(x + a + sign * nx, REAR_Y + offset, z + sign * nz)
                             for a, z, sign in [(left[0], left[1], -1), (right[0], right[1], -1),
                                                (right[0], right[1], 1), (left[0], left[1], 1)]])
            polygons.append(tuple(range(first, first + 4)))
            materials.append(index)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], polygons)
    mesh.materials.append(edge)
    mesh.materials.append(mat)
    for polygon, index in zip(mesh.polygons, materials):
        polygon.material_index = index
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.visible_shadow = False
    return obj, [(x + a, REAR_Y, z) for a, z in coordinates]


def aim(camera, location, target, scale):
    camera.location = location
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = scale
    camera.data.clip_start = .01
    camera.data.clip_end = 100
    bpy.context.view_layer.update()


def box(scene, camera, points):
    coords = [world_to_camera_view(scene, camera, Vector(p)) for p in points]
    width, height = scene.render.resolution_x, scene.render.resolution_y
    xs, ys = [c.x * width for c in coords], [(1 - c.y) * height for c in coords]
    return [max(0, math.floor(min(xs))), max(0, math.floor(min(ys))),
            min(width, math.ceil(max(xs))), min(height, math.ceil(max(ys)))]


def panel_points(x, y):
    return [(x + a, y, z) for a in (-.52, .52) for z in (BOTTOM, TOP)]


def visibility(scene, camera, obj, x, y):
    """Parallel camera rays audit exposed surfaces, independently of image IDs."""
    graph = bpy.context.evaluated_depsgraph_get()
    forward = camera.rotation_euler.to_matrix() @ Vector((0, 0, -1))
    visible, total = 0, 25
    for dx in np.linspace(-.45, .45, 5):
        for z in np.linspace(BOTTOM + .05, TOP - .05, 5):
            point = Vector((x + float(dx), y, float(z)))
            display = world_to_camera_view(scene, camera, point)
            if not (0 <= display.x <= 1 and 0 <= display.y <= 1 and display.z > 0):
                continue
            origin = point - forward * (point - camera.location).dot(forward)
            hit, location, _, _, first, _ = scene.ray_cast(
                graph, origin, forward, distance=(point - origin).length + .03)
            if hit and first.name == obj.name and (location - point).length < .025:
                visible += 1
    return {"visible": visible, "sampled": total}


def mask_output(scene, directory):
    """Render first-hit semantic surfaces; no guessed visibility flags."""
    scene.view_layers[0].use_pass_object_index = True
    scene.view_layers[0].use_pass_material_index = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    combine = nodes.new("CompositorNodeCombRGBA")
    for index, channel in enumerate(("R", "G", "B"), 1):
        mask = nodes.new("CompositorNodeIDMask")
        mask.index = index
        mask.use_antialiasing = False
        scene.node_tree.links.new(layers.outputs["IndexMA" if index == 3 else "IndexOB"], mask.inputs[0])
        scene.node_tree.links.new(mask.outputs[0], combine.inputs[channel])
    output = nodes.new("CompositorNodeOutputFile")
    output.base_path = str(directory)
    output.format.file_format = "PNG"
    output.format.color_mode = "RGB"
    output.format.color_depth = "8"
    output.file_slots[0].save_as_render = False
    scene.node_tree.links.new(combine.outputs[0], output.inputs[0])
    composite = nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layers.outputs["Image"], composite.inputs[0])
    return output


def configure(seed, samples):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_light_tree = False
    scene.render.dither_intensity = 0
    scene.cycles.seed = seed
    scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.transparent_max_bounces = 0
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.65, .65, .65, 1)
    background = scene.world.node_tree.nodes["Background"]
    background.inputs[1].default_value = 1
    light_path = scene.world.node_tree.nodes.new("ShaderNodeLightPath")
    scene.world.node_tree.links.new(light_path.outputs["Is Camera Ray"], background.inputs[1])
    device = "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for candidate in prefs.devices:
            candidate.use = candidate.type == "CUDA"
        if any(d.use for d in prefs.devices):
            scene.cycles.device = "GPU"
            device = "CUDA"
    except (TypeError, RuntimeError):
        scene.cycles.device = "CPU"
    bpy.ops.object.light_add(type="SUN", location=(0, -3, 6))
    light = bpy.context.object
    light.data.energy, light.data.angle = 2.0, 0
    light.rotation_euler = (Vector((0, 0, .78)) - light.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.object.light_add(type="SUN", location=(0, 2, 8))
    light = bpy.context.object
    light.data.energy, light.data.angle = 3.5, 0
    light.rotation_euler = (Vector((0, 0, .78)) - light.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.object.camera_add()
    scene.camera = bpy.context.object
    return scene, device


def render_bank(output, seed=42, samples=64, device="CPU"):
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; preserve existing rendered banks")
    assets = output / "assets"
    assets.mkdir(parents=True)
    masks = output / "masks"
    masks.mkdir()
    scene, available_device = configure(seed, samples)
    if device == "CPU":
        scene.cycles.device = "CPU"
    elif available_device != "CUDA":
        raise RuntimeError("CUDA rendering requested but unavailable")
    mask_node = mask_output(scene, masks)
    rng = random.Random(seed)
    wood = material("common opaque wood", (.38, .235, .12))
    rear = material("common rear panel", (.42, .27, .14))
    shield = material("inspection rack guards", (.30, .32, .33))
    floor = material("neutral floor", (.42, .43, .44))
    exposed = material("exposed wood scratch", (.68, .51, .30))
    exposed.pass_index = 3
    identity_seeds = [rng.randrange(1, 2**30) for _ in SLOTS]
    identities = [wood_material(s) for s in identity_seeds]
    fronts, rears, scratches, scratch_points = [], [], [], []
    paths = []
    for index, x in enumerate(CENTERS):
        cube(f"opaque-box-{index}", (x, 0, .78), (1.08, .90, 1.16), wood)
        fronts.append(face(f"front-{index}", x, FRONT_Y, identities[index]))
        rears.append(face(f"rear-{index}", x, REAR_Y, rear))
        defect_rng = random.Random(f"{seed}|scratch|{index}")
        angle = defect_rng.uniform(-.3, .3)
        tx, tz = defect_rng.uniform(-.08, .08), defect_rng.uniform(.62, .86)
        coordinates = []
        for a, z in [(-.24, -.08), (-.13, -.01), (-.02, -.03), (.13, .08), (.25, .11)]:
            a, z = a + defect_rng.uniform(-.02, .02), z + defect_rng.uniform(-.02, .02)
            coordinates.append((tx + a * math.cos(angle) - z * math.sin(angle),
                                tz + a * math.sin(angle) + z * math.cos(angle)))
        paths.append(coordinates)
        obj, points = scratch(f"scratch-{index}", x, coordinates, exposed)
        scratches.append(obj)
        fronts[-1].pass_index, rears[-1].pass_index, obj.pass_index = 1, 2, 3
        scratch_points.append(points)
        cube(f"guard-{index}", (x, 1.75, .90), (1.26, .08, 1.90), shield)
    cube("floor", (0, 0, -.05), (30, 30, .1), floor)
    edge_materials = [obj.data.materials[0] for obj in scratches]
    images = {}
    started = time.monotonic()

    def expose(permutation=(0, 1, 2), state=(0, 0, 0)):
        for slot, identity in enumerate(permutation):
            fronts[slot].data.materials[0] = identities[identity]
            # Keep the BVH identical: condition changes only visible surface material.
            scratches[slot].data.materials[0] = edge_materials[slot] if state[slot] else rear
            scratches[slot].data.materials[1] = exposed if state[slot] else rear
            scratches[slot].pass_index = 3 if state[slot] else 2
        bpy.context.view_layer.update()

    def save(key, location, target, scale, dimensions, permutation, state, kind, slot=None):
        expose(permutation, state)
        scene.render.resolution_x, scene.render.resolution_y = dimensions
        aim(scene.camera, location, target, scale)
        path = assets / (key + ".png")
        scene.render.filepath = str(path)
        mask_node.file_slots[0].path = key
        bpy.ops.render.render(write_still=True)
        mask_path = masks / (key + ".png")
        (masks / (key + f"{scene.frame_current:04d}.png")).rename(mask_path)
        mask_image = bpy.data.images.load(str(mask_path), check_existing=False)
        pixels = np.empty(len(mask_image.pixels), dtype=np.float32)
        mask_image.pixels.foreach_get(pixels)
        pixels = pixels.reshape(dimensions[1], dimensions[0], 4)
        counts = {name: int(np.count_nonzero(pixels[:, :, i] > .5))
                  for i, name in enumerate(("front", "rear", "scratch"))}
        bpy.data.images.remove(mask_image)
        record = {"path": str(path.relative_to(output)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "size": list(dimensions), "kind": kind, "permutation": list(permutation),
                  "states": list(state), "camera": {"location": list(location), "target": list(target),
                  "orthographic_scale": scale}, "boxes": {}, "visibility": {},
                  "visibility_checks": {"mask_path": str(mask_path.relative_to(output)),
                     "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                     "pixel_counts": counts, "method": "Cycles first-hit object/material-index RGB masks"}}
        for i, address in enumerate(SLOTS):
            record["visibility"][address] = {
                "front": visibility(scene, scene.camera, fronts[i], CENTERS[i], FRONT_Y),
                "rear": visibility(scene, scene.camera, rears[i], CENTERS[i], REAR_Y)}
        if kind in ("reference", "front_survey"):
            chosen = [1] if kind == "reference" else range(3)
            front_boxes = {SLOTS[i]: box(scene, scene.camera, panel_points(CENTERS[i], FRONT_Y))
                           for i in chosen}
            record["boxes"] = {"front": front_boxes["center"]} if kind == "reference" else front_boxes
        if slot is not None:
            panel = box(scene, scene.camera, panel_points(CENTERS[slot], REAR_Y))
            record["boxes"] = {"panel": panel, "object": panel,
                               "defect": [max(0, v - 4) if i < 2 else min(dimensions[i % 2], v + 4) for i, v in enumerate(box(scene, scene.camera, scratch_points[slot]))] if state[slot] else None}
        rgb = bpy.data.images.load(str(path), check_existing=False)
        rgb.colorspace_settings.name = "Non-Color"
        values = np.empty(len(rgb.pixels), dtype=np.float32)
        rgb.pixels.foreach_get(values)
        decoded = np.rint(values.reshape(dimensions[1], dimensions[0], 4)[::-1, :, :3] * 255).astype(np.uint8)
        record["decoded_rgb_sha256"] = hashlib.sha256(decoded.tobytes()).hexdigest()
        bpy.data.images.remove(rgb)
        images[key] = record
        print(json.dumps({"rendered": key, "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)

    save("reference", (0, -3, .78), (0, 0, .78), 1.38, (512, 512),
         (1, 0, 2), (0, 0, 0), "reference")
    save("initial", (0, 6, 2.8), (0, 0, .78), 6.2, (768, 512),
         (0, 1, 2), (0, 0, 0), "initial")
    for index, permutation in enumerate(itertools.permutations(range(3))):
        key = f"front_permutation_{index}"
        save(key, (0, -6, 2.5), (0, 0, .78), 6.2, (768, 512),
             permutation, (0, 0, 0), "front_survey")
    for slot, address in enumerate(SLOTS):
        for bit in (0, 1):
            state = [0, 0, 0]
            state[slot] = bit
            save(f"rear_{address}_{'scratched' if bit else 'intact'}", (CENTERS[slot], 1.35, .78),
                 (CENTERS[slot], 0, .78), 1.40, (512, 512),
                 (0, 1, 2), state, "rear_detail", slot)
    audits = []
    controls = [
        ("initial", (2, 1, 0), (1, 1, 1), None),
        ("front_permutation_0", (0, 1, 2), (1, 1, 1), None),
        ("rear_left_intact", (2, 1, 0), (0, 1, 1), 0)]
    for index, (base, permutation, state, slot) in enumerate(controls):
        original = images[base]
        camera = original["camera"]
        key = f"audit-{index}"
        save(key, camera["location"], camera["target"], camera["orthographic_scale"],
             tuple(original["size"]), permutation, state, original["kind"], slot)
        alternative = images.pop(key)
        equal = original["decoded_rgb_sha256"] == alternative["decoded_rgb_sha256"]
        audits.append({"base_image": base, "base_parameters": {"permutation": original["permutation"],
                       "states": original["states"]}, "alternative_parameters":
                       {"permutation": list(permutation), "states": list(state)},
                       "base_rgb_sha256": original["decoded_rgb_sha256"],
                       "alternative_rgb_sha256": alternative["decoded_rgb_sha256"], "equal": equal})
        if not equal:
            raise RuntimeError(f"Counterfactual image independence failed: {base}; retain audit files")
        (output / alternative["path"]).unlink()
        (output / alternative["visibility_checks"]["mask_path"]).unlink()
    for key, record in images.items():
        counts = record["visibility_checks"]["pixel_counts"]
        if key == "initial":
            assert counts == {"front": 0, "rear": 0, "scratch": 0}, counts
        elif record["kind"] in ("reference", "front_survey"):
            assert counts["front"] > 100 and counts["rear"] == counts["scratch"] == 0, counts
        else:
            assert counts["front"] == 0 and counts["rear"] > 100, counts
            assert (counts["scratch"] > 0) == key.endswith("scratched"), counts
    metadata = {"schema": "resolveai-factorized-bank-v1", "seed": seed, "images": images,
                "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "slots": list(SLOTS), "identity_seeds": identity_seeds,
                "target_identity": 0, "private_scratch_paths": paths,
                "decoded_rgb_order": "top-to-bottom rows, left-to-right pixels, uint8 RGB",
                "permutations": list(itertools.permutations(range(3))),
                "states": list(itertools.product((0, 1), repeat=3)),
                "private_mesh_ids": {"front": [o.name for o in fronts], "rear": [o.name for o in rears],
                                     "scratch": [o.name for o in scratches]},
                "counterfactual_rgb_checks": audits,
                "render_settings": {"blender": bpy.app.version_string, "engine": "Cycles", "device": device,
                                    "samples": samples, "max_bounces": 1, "diffuse_bounces": 0,
                                    "glossy_bounces": 0, "denoising": False, "adaptive_sampling": False,
                                    "dither": 0, "light_tree": False, "lighting": "deterministic matte normal shading; no stochastic reflections", "scratch_shadows": False},
                "scope": "Synthetic static-scene factors; not independent human annotation or real-photo evaluation",
                "elapsed_s": round(time.monotonic() - started, 2)}
    (output / "bank.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", choices=["CPU", "CUDA"], default="CPU")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.samples < 1:
        parser.error("samples must be positive")
    result = render_bank(args.output, args.seed, args.samples, args.device)
    print(json.dumps({"complete": True, "images": len(result["images"]), "elapsed_s": result["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
