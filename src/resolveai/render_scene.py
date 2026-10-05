"""Blender state-driven smoke scene; run with Blender --background --python.

Generated geometry labels are provisional, not independently reviewed dataset
annotations. Requests later release these pixels without changing scene state.
"""
import argparse
import json
from pathlib import Path
import random
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def material(name, color):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1)
    return value


def cube(name, location, dimensions, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    obj.data.materials.append(mat)
    return obj


def point_camera(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def projected_box(scene, camera, points, resolution):
    coords = [world_to_camera_view(scene, camera, Vector(p)) for p in points]
    xs, ys = [c.x*resolution for c in coords], [(1-c.y)*resolution for c in coords]
    return [max(0,int(min(xs))-2),max(0,int(min(ys))-2),
            min(resolution,int(max(xs))+3),min(resolution,int(max(ys))+3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.output.mkdir(parents=True,exist_ok=True)
    assets=args.output/"assets";assets.mkdir(exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.render.engine="CYCLES"
    scene.cycles.samples=16
    scene.cycles.use_denoising=True
    scene.render.resolution_x=scene.render.resolution_y=512
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    device="CPU"
    try:
        prefs=bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type="CUDA"
        prefs.get_devices()
        for candidate in prefs.devices:
            candidate.use=candidate.type=="CUDA"
        if any(d.use for d in prefs.devices):
            scene.cycles.device="GPU"
            device="CUDA"
    except (TypeError,RuntimeError):
        scene.cycles.device="CPU"
    rng=random.Random(args.seed)
    wood=material("natural wood",(.36,.18,.07))
    wood.use_nodes=True
    shader=wood.node_tree.nodes.get("Principled BSDF")
    noise=wood.node_tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value=13+rng.random()*3
    coordinates=wood.node_tree.nodes.new("ShaderNodeTexCoord")
    mapping=wood.node_tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value=(1,8,1)
    ramp=wood.node_tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color=(.12,.045,.012,1)
    ramp.color_ramp.elements[1].color=(.46,.25,.10,1)
    wood.node_tree.links.new(coordinates.outputs["Generated"],mapping.inputs["Vector"])
    wood.node_tree.links.new(mapping.outputs["Vector"],noise.inputs["Vector"])
    wood.node_tree.links.new(noise.outputs["Fac"],ramp.inputs["Fac"])
    wood.node_tree.links.new(ramp.outputs["Color"],shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value=.68
    table=cube("tabletop",(0,0,.8),(1.8,1.2,.06),wood)
    for x in (-.75,.75):
        for y in (-.45,.45): cube("leg",(x,y,.39),(.09,.09,.78),wood)
    cube("floor",(0,0,-.06),(8,8,.1),material("neutral floor",(.34,.34,.34)))
    scratch_mat=material("exposed scratch",(.65,.47,.27))
    scratch_points=[(-.22,.08,.832),(-.11,.105,.832),(.025,.06,.832),(.16,.09,.832)]
    curve=bpy.data.curves.new("scratch geometry",type="CURVE")
    curve.dimensions="3D"
    curve.bevel_depth=.004
    curve.bevel_resolution=1
    spline=curve.splines.new("POLY");spline.points.add(len(scratch_points)-1)
    for point,xyz in zip(spline.points,scratch_points): point.co=(*xyz,1)
    scratch=bpy.data.objects.new("scratch",curve)
    scene.collection.objects.link(scratch);scratch.data.materials.append(scratch_mat)
    bpy.ops.object.light_add(type="AREA",location=(1,-2,4))
    light=bpy.context.object;light.data.energy=650;light.data.shape="DISK";light.data.size=4
    point_camera(light,(0,0,.6))
    scene.world.color=(.25,.25,.25)
    bpy.ops.object.camera_add();camera=bpy.context.object;scene.camera=camera
    views={"front":(2.4,-3.2,2.7),"side":(2.5,1.9,2.5),"detail":(.45,-.65,1.55)}
    evidence=[];regions={};object_regions={}
    for time in ("before","after"):
        scratch.hide_render=time=="before"
        for view,position in views.items():
            camera.location=position
            camera.data.lens=48 if view!="detail" else 45
            point_camera(camera,(0,0,.8))
            bpy.context.view_layer.update()
            image_id=time+"-"+view
            scene.render.filepath=str(assets/(image_id+".png"))
            bpy.ops.render.render(write_still=True)
            evidence.append({"id":image_id,"path":"assets/"+image_id+".png",
                "source_id":image_id,"party":"A" if rng.random()<.5 else "B",
                "object":"table","time":time,"view":view,"available":True})
            points=[tuple(table.matrix_world @ v.co) for v in table.data.vertices] if time=="before" else scratch_points
            regions[image_id]=projected_box(scene,camera,points,512)
            object_regions[image_id]=projected_box(scene,camera,[tuple(table.matrix_world @ v.co) for v in table.data.vertices],512)
    family=f"blender-table-{args.seed}"
    scene_state={"scene_id":family,"asset_id":f"table-{args.seed}",
                 "objects":[{"identity":f"table-{args.seed}","public_name":"table",
                             "conditions":{"before":"intact","after":"scratched"}}],
                 "views":views,"renderer":"Blender 4.3.2 Cycles"}
    def region(image_id, box):
        return {"image_id":image_id,"time":image_id.split("-")[0],"bbox":box}
    endpoints={i:region(i,object_regions[i]) for i in ("before-front","after-detail")}
    identity={"relation":"same_object","left":endpoints["before-front"],"right":endpoints["after-detail"]}
    order={**identity,"relation":"earlier_than"}
    subclaims=[
        {"id":"identity","kind":"identity","truth":"Supported","minimal_evidence_sets":{"Supported":[[identity]]}},
        {"id":"before","kind":"state","truth":"Supported","minimal_evidence_sets":{"Supported":[[region("before-front",regions["before-front"])]]}},
        {"id":"after","kind":"state","truth":"Supported","minimal_evidence_sets":{"Supported":[[region("after-detail",regions["after-detail"])]]}},
        {"id":"order","kind":"time","truth":"Supported","minimal_evidence_sets":{"Supported":[[order]]}}]
    annotation={"verdict":"Supported","status":"unreviewed","protocol":"evidence-chain-v1",
                "annotation_origin":"provisional procedural geometry, not independent review","subclaims":subclaims}
    parts=[{"id":"identity","text":"The two pictures show the same table."},
           {"id":"before","text":"The visible tabletop region was unmarked before."},
           {"id":"after","text":"That region has a scratch after."},
           {"id":"order","text":"The unmarked photograph precedes the scratched photograph."}]
    cases=[]
    for variant in ("Sufficient","Obtainable","Missing","Unavailable"):
        rows=[dict(e) for e in evidence]
        if variant=="Missing": rows=[e for e in rows if e["id"]!="after-detail"]
        if variant=="Unavailable":
            for row in rows:
                if row["id"]=="after-detail": row["available"]=False
        cases.append({"case_id":family+"-"+variant,"family_id":family,
            "variant":variant,"scene":scene_state,
            "claim":"The same table has an unmarked visible tabletop region before and a scratch in that region after.",
            "claim_parts":parts,
            "initial":["before-front","after-detail" if variant=="Sufficient" else "after-front"],
            "evidence":rows,"request_options":{"objects":["table"],"times":["before","after"],"views":list(views)},
            "annotation":annotation})
    (args.output/"cases.json").write_text(json.dumps(cases,indent=2)+"\n")
    manifest={"renderer":"Blender 4.3.2","device":device,"images":len(evidence),
              "seed":args.seed,"annotation_status":"unreviewed",
              "scope":"state/camera/provenance smoke only; not research dataset validation"}
    (args.output/"render_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest),flush=True)


if __name__=="__main__": main()
