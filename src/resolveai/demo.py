"""Produce a Chinese GitHub operation demo from actual rendered pixels/tools."""
import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .conversation import Conversation
from .environment import Environment
from .grounding_tools import FrozenGrounding


def public_result(result):
    result = deepcopy(result)
    for image in result.get("images", []):
        image["image_sha256"] = hashlib.sha256(image.pop("image_png")).hexdigest()
    return result


def draw_boxes(image, proposals):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for row in proposals:
        if "bbox" in row:
            draw.rectangle(row["bbox"], outline="#1967d2", width=3)
    return image


def sheet(images, destination, columns=4, font_path=None):
    width, height = 320, 290
    canvas = Image.new("RGB", (columns*width, ((len(images)+columns-1)//columns)*height), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(font_path) if font_path else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    for i, (label, image) in enumerate(images):
        x, y = (i%columns)*width, (i//columns)*height
        image = ImageOps.contain(image, (width-16, height-42))
        canvas.paste(image, (x+(width-image.width)//2, y+36+(height-42-image.height)//2))
        draw.text((x+12,y+8), label, font=font, fill="#111827")
    canvas.save(destination, "WEBP", quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("grounding", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--font", type=Path, required=True, help="Chinese-capable font for demo captions")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "assets"
    assets.mkdir(exist_ok=True)
    case = next(c for c in json.loads((args.data/"cases.json").read_text()) if c["variant"] == "Obtainable")
    case = deepcopy(case)
    # Separate, explicitly synthetic OCR card; never a damage/identity label.
    fixture = args.data / "demo-note.png"
    note = Image.new("RGB", (600,160), "white")
    draw = ImageDraw.Draw(note)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",40)
    draw.text((20,40), "BEFORE 2026 TABLE", font=font, fill="black")
    note.save(fixture)
    case["evidence"].append({"id":"note","path":fixture.name,"source_id":"synthetic-ocr-card",
        "party":"demo","object":"note","time":"capture","view":"label","available":True})
    case["initial"].append("note")
    session = Conversation(Environment(case,args.data,30,grounding_backend=FrozenGrounding(args.grounding)))
    rows, panels = [], []
    def call(name, arguments):
        action = {"name":name,"arguments":arguments}
        result = session.call(action)
        if "error" in result:
            raise RuntimeError(result["error"])
        rows.append({"action":action,"result":public_result(result)})
        return result
    def pixels(result, index=0):
        return Image.open(BytesIO(result["images"][index]["image_png"])).convert("RGB")
    base = call("inspect",{"image_id":"after-front"})
    panels.append(("A 检查原图 512×512",pixels(base)))
    enlarged = call("zoom",{"image_id":"after-front","factor":2})
    panels.append(("B 放大 1024×1024",pixels(enlarged)))
    reduced = call("zoom",{"image_id":"after-front","factor":.5})
    panels.append(("C 缩小 256×256",pixels(reduced)))
    cropped = call("crop",{"image_id":"after-front","bbox":[210,220,320,310]})
    panels.append(("D 裁剪 110×90",pixels(cropped)))
    ocr = call("ocr",{"image_id":"note"})
    annotated = note.copy()
    draw = ImageDraw.Draw(annotated)
    for row in ocr["text_regions"]:
        draw.line([tuple(p) for p in row["polygon"]]+[tuple(row["polygon"][0])],fill="#1967d2",width=3)
    panels.append(("E OCR 文字识别",annotated))
    text_image = call("ground_text_to_image",{"image_id":"after-front","text":"table"})
    panels.append(("F 文字 → 图像区域",draw_boxes(pixels(base),text_image["proposals"])))
    image_text = call("ground_image_to_text",{"image_id":"after-front",
        "candidates":["a wooden table","a bottle","a chair"]})
    ranking = Image.new("RGB",(320,230),"white")
    draw = ImageDraw.Draw(ranking)
    small = ImageFont.truetype(str(args.font) if args.font else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",17)
    text_labels = {"a wooden table": "木桌", "a bottle": "瓶子", "a chair": "椅子"}
    for i,row in enumerate(image_text["proposals"]):
        draw.text((12,30+i*55),f"{text_labels[row['text']]}：{row['similarity']:.3f}",font=small,fill="black")
    panels.append(("G 图像 → 候选文字",ranking))
    requested = call("request_photo",{"query":{"object":"table","time":"after","view":"detail"}})
    panels.append(("H 请求新视角照片",pixels(requested)))
    detail = call("crop",{"image_id":"after-detail","bbox":[140,150,380,280]})
    panels.append(("I 裁剪新视角 240×130",pixels(detail)))
    matching = call("ground_image_to_image",{"query_image_id":detail["images"][0]["view_id"],
        "target_image_id":"after-front","top_k":2})
    panels.append(("J 图像 → 图像对应",draw_boxes(pixels(base),matching["proposals"])))
    compared = call("compare",{"image_ids":["before-front","after-detail"]})
    pair = Image.new("RGB",(1024,512),"white")
    pair.paste(pixels(compared,0),(0,0)); pair.paste(pixels(compared,1),(512,0))
    panels.append(("K 前后照片对比",pair))
    quality = call("assess_quality",{"image_id":"after-front"})
    metadata = call("read_metadata",{"image_id":"after-front"})
    numbers = Image.new("RGB",(320,230),"white")
    draw = ImageDraw.Draw(numbers)
    lines=["质量检查与来源元数据",f"平均亮度：{quality['luminance_mean']:.1f}",
        f"对比度：{quality['contrast_std']:.1f}",f"边缘方差：{quality['edge_variance']:.1f}",
        "提交时点："+metadata["time"],"文件哈希："+metadata["file_sha256"][:16]]
    for i,line in enumerate(lines):draw.text((12,20+30*i),line,font=small,fill="black")
    panels.append(("L 质量检查 / 元数据",numbers))
    call("finish",{"verdict":"Supported","citations":[
        {"image_id":"before-front","bbox":[0,0,512,512],"time":"before"},
        {"image_id":"after-detail","bbox":[140,150,380,280],"time":"after"}]})
    failed=[]
    for variant in ("Missing","Unavailable"):
        private=next(c for c in json.loads((args.data/"cases.json").read_text()) if c["variant"]==variant)
        branch=Conversation(Environment(private,args.data,12))
        failed.append({"variant":variant,"action":{"name":"request_photo","arguments":{
            "query":{"object":"table","time":"after","view":"detail"}}},
            "result":public_result(branch.call({"name":"request_photo","arguments":{
            "query":{"object":"table","time":"after","view":"detail"}}}))})
    sheet(panels,assets/"operations.webp",font_path=args.font)
    views=[({"before-front":"之前 · 正面","after-front":"之后 · 正面",
             "after-side":"之后 · 侧面","after-detail":"之后 · 细节"}[image_id],Image.open(args.data/"assets"/(image_id+".png")).convert("RGB"))
           for image_id in ["before-front","after-front","after-side","after-detail"]]
    sheet(views,assets/"views.webp",columns=2,font_path=args.font)
    manifest={"scope":"scripted interface demonstration; not a model accuracy result",
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "renderer":json.loads((args.data/"render_manifest.json").read_text()),
        "providers":json.loads((args.grounding/"manifest.json").read_text()),
        "budget":30,"operations":len(rows),"model_generated_actions":False,
        "ocr_fixture":"programmatically printed card; no real collection or annotation claim"}
    (args.output/"trace.json").write_text(json.dumps({"manifest":manifest,"operations":rows,
        "failed_requests":failed},indent=2)+"\n")
    print(json.dumps({"operations":len(rows),"ocr":ocr["text_regions"],
        "image_text":image_text["proposals"],"request":requested["status"]}),flush=True)


if __name__=="__main__":main()
