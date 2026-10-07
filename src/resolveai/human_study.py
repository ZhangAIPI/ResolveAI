"""Short, blinded web study. Pixels and actions use the actual evidence environment."""

import argparse
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
from threading import RLock
import time
from urllib.parse import unquote

from PIL import Image
from .environment import Environment
from .review import validate_truth_vote
from .tools import ActionError, VERDICTS

UI_PROTOCOL = "human-ui-v5-optional-regions"
TOOLS = {"inspect", "crop", "zoom", "compare", "ocr", "request_photo", "finish"}
CATEGORIES = {
    "bottle": "瓶子",
    "cable": "电缆",
    "capsule": "胶囊",
    "hazelnut": "榛子",
    "metal_nut": "螺母",
    "pill": "药片",
    "screw": "螺丝",
    "toothbrush": "牙刷",
    "transistor": "晶体管",
    "zipper": "拉链",
}
CONDITIONS = {
    "a chipped or broken rim": "瓶口有缺口或破损",
    "visible foreign material or contamination": "有可见异物或污染",
    "a bent wire": "电线弯曲",
    "a wire arrangement inconsistent with the normal reference": "电线排列与正常参考不一致",
    "cut outer insulation": "外绝缘层有切口",
    "a missing cable": "缺少电缆",
    "a missing wire": "缺少一根电线",
    "a puncture in the insulation": "绝缘层有刺孔",
    "a crack": "有裂纹",
    "a faulty printed marking": "印字有异常",
    "a puncture": "有刺孔",
    "a surface scratch": "表面有划痕",
    "a squeezed or deformed body": "主体被挤压或变形",
    "a cut": "有切口",
    "a hole": "有孔洞",
    "an abnormal surface marking": "表面有异常印记",
    "a bent or deformed body": "主体弯曲或变形",
    "abnormal discoloration": "有异常变色",
    "an orientation inconsistent with the normal reference": "朝向与正常参考不一致",
    "visible contamination": "有可见污染",
    "an appearance inconsistent with the normal pill reference": "外观与正常药片参考不一致",
    "a deformed tip": "尖端变形",
    "a scratch on the head": "头部有划痕",
    "a scratch on the neck": "颈部有划痕",
    "damaged threading on the side": "侧面螺纹受损",
    "damaged threading near the tip": "尖端附近螺纹受损",
    "a bristle arrangement inconsistent with the normal reference": "刷毛排列与正常参考不一致",
    "a bent lead": "引脚弯曲",
    "a cut lead": "引脚断裂",
    "a damaged case": "外壳受损",
    "a placement inconsistent with the normal reference": "摆放位置与正常参考不一致",
    "broken teeth": "齿有破损",
    "damage to the fabric border": "布边受损",
    "rough or damaged teeth": "齿粗糙或受损",
    "split teeth": "齿开裂",
    "deformed teeth": "齿变形",
}


def chinese_claim(case):
    if case["task"] == "identity":
        return "照片中的对象 A 与对象 B，是同一件实物吗？"
    return (
        "声明：目标"
        + CATEGORIES[case["category"]]
        + CONDITIONS[case["provenance"]["criterion"]]
        + "。"
    )


class Study:
    def __init__(self, root):
        self.root = Path(root)
        self.plan = json.loads((self.root / "plan.json").read_text())
        data = Path(self.plan["data"])
        if (
            hashlib.sha256((data / "cases.json").read_bytes()).hexdigest()
            != self.plan["dataset_sha256"]
        ):
            raise ValueError("Study dataset changed")
        self.cases = {
            c["case_id"]: c
            for c in json.loads((data / "cases.json").read_text())
            if c["family_id"] in self.plan["family_ids"]
        }
        self.access = json.loads((self.root / "access.json").read_text())
        self.db = sqlite3.connect(
            self.root / "responses.sqlite", check_same_thread=False
        )
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS consent(actor TEXT PRIMARY KEY, at REAL);
            CREATE TABLE IF NOT EXISTS starts(actor TEXT, idx INTEGER, at REAL, PRIMARY KEY(actor,idx));
            CREATE TABLE IF NOT EXISTS events(actor TEXT, idx INTEGER, seq INTEGER, payload TEXT, PRIMARY KEY(actor,idx,seq));
            CREATE TABLE IF NOT EXISTS answers(actor TEXT, idx INTEGER, payload TEXT, PRIMARY KEY(actor,idx));
        """)
        self.db.commit()
        self.lock = RLock()
        self.environments = {}
        self.pictures = {}

    def actor(self, token):
        if not token or token not in self.access:
            raise PermissionError("Invalid invitation")
        return self.access[token]

    def current(self, actor):
        done = {
            r[0]
            for r in self.db.execute(
                "SELECT idx FROM answers WHERE actor=?", (actor["id"],)
            )
        }
        return next((i for i in range(len(actor["tasks"])) if i not in done), None)

    def events(self, actor, index):
        return [
            json.loads(r[0])
            for r in self.db.execute(
                "SELECT payload FROM events WHERE actor=? AND idx=? ORDER BY seq",
                (actor["id"], index),
            )
        ]

    def append(self, actor, index, payload):
        count = len(self.events(actor, index))
        with self.db:
            self.db.execute(
                "INSERT INTO events VALUES (?,?,?,?)",
                (actor["id"], index, count, json.dumps(payload, ensure_ascii=False)),
            )

    def case(self, task):
        return self.cases[task.get("case_id", task["family_id"] + "-Obtainable")]

    def environment(self, actor, index):
        key = (actor["id"], index)
        if key not in self.environments:
            task = actor["tasks"][index]
            case = self.case(task)
            env = Environment(case, case["asset_root"], 0)
            if task.get("condition", "full_available") == "full_available":
                env._released.update(
                    e["id"] for e in case["evidence"] if e["available"]
                )
            for event in self.events(actor, index):
                if event.get("action") and not event.get("presentation_only"):
                    try:
                        self.human_step(env, event["action"])
                    except ActionError:
                        pass
            self.environments[key] = env
        return self.environments[key]

    @staticmethod
    def human_step(env, action):
        """Keep real tools and counters; fund each human action without a cap."""
        env.budget = getattr(env.costs, action.get("type", ""), 0)
        return env.step(action)

    def elapsed(self, actor, index):
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO starts VALUES (?,?,?)",
                (actor["id"], index, time.time()),
            )
        started = self.db.execute(
            "SELECT at FROM starts WHERE actor=? AND idx=?", (actor["id"], index)
        ).fetchone()[0]
        return max(0, time.time() - started)

    def visible(self, actor, index):
        env = self.environment(actor, index)
        ids = [i for i in env._release_order if i in env._released]
        ids.extend(i for i in env._evidence if i in env._released and i not in ids)
        for event in self.events(actor, index):
            for ref in event.get("views", []):
                if ref not in ids:
                    ids.append(ref)
        pictures = []
        for ref in ids:
            key = (actor["id"], index, ref)
            if key not in self.pictures:
                picture = env._image(ref)
                self.pictures[key] = {**picture, "bytes": picture["image_png"]}
            pictures.append(self.pictures[key])
        return pictures

    def state(self, token):
        actor = self.actor(token)
        if actor["role"] == "admin":
            return self.admin()
        if not self.db.execute(
            "SELECT 1 FROM consent WHERE actor=?", (actor["id"],)
        ).fetchone():
            return {"role": "participant", "intro": True, "actor": actor["id"]}
        index = self.current(actor)
        if index is None:
            return {"done": True, "actor": actor["id"]}
        task = actor["tasks"][index]
        case = self.case(task)
        elapsed = self.elapsed(actor, index)
        images = self.visible(actor, index)
        for picture in images:
            self.pictures[(actor["id"], index, picture["view_id"])] = picture
        result = {
            "actor": actor["id"],
            "index": index,
            "total": len(actor["tasks"]),
            "mode": task["mode"],
            "claim": chinese_claim(case),
            "original_claim": case["claim"],
            "task": case["task"],
            "elapsed_seconds": round(elapsed, 1),
            "ui_protocol": UI_PROTOCOL,
            "images": [
                {k: v for k, v in p.items() if k not in {"bytes", "image_png"}}
                for p in images
            ],
            "context": "正常参考来自另一个实物，不是目标的之前照片。"
            if case["task"] == "state"
            else "A、B 只是提交对象槽位。相似不等于同一实物；不要只依据背景判断。",
        }
        env = self.environment(actor, index)
        by_view = {p["view_id"]: p for p in images}
        displayed = {}
        scales = {}
        for picture in images:
            if picture["image_id"] != picture["view_id"]:
                continue
            key = picture["source_id"]
            old = by_view.get(displayed.get(key))
            if old is None or (
                picture["display_size"][0] * picture["display_size"][1]
                > old["display_size"][0] * old["display_size"][1]
            ):
                displayed[key] = picture["view_id"]
        for event in self.events(actor, index):
            action = event.get("action", {})
            if event.get("presentation_only"):
                picture = by_view[action["image_id"]]
                key = picture["source_id"]
                scales[key] = min(4, max(0.25, scales.get(key, 1) * action["factor"]))
            elif action.get("type") in {"inspect", "crop"}:
                for ref in event.get("views", []):
                    if ref in by_view:
                        key = by_view[ref]["source_id"]
                        displayed[key] = ref
                        scales[key] = 1
        for picture in result["images"]:
            picture["display_scale"] = scales.get(picture["source_id"], 1)
        result.update(
            display_views=list(displayed.values()),
            condition=task.get("condition", "full_available"),
            turns=sum(bool(e.get("action")) for e in self.events(actor, index)),
            requests=env.requests,
            request_options=env._case.get("request_options", {}),
            ocr_available=env._ocr.available,
        )
        if task["mode"] == "review":
            result["review_scope"] = "full_pool_only"
        return result

    def selections(self, images, selected, verdict, task):
        rows = {p["view_id"]: p for p in images}
        if not isinstance(selected, list):
            raise ValueError("画框记录格式错误")
        citations = []
        objects = set()
        for item in selected:
            picture = rows[item["view_id"]]
            box = item["bbox"]
            Environment._check_box(box, picture["source_size"], "source_pixels")
            b = picture["source_bbox"]
            if not (
                b[0] <= box[0] < box[2] <= b[2] and b[1] <= box[1] < box[3] <= b[3]
            ):
                raise ValueError("证据框超出可见图片范围")
            citations.append(
                {"image_id": picture["image_id"], "bbox": box, "time": picture["time"]}
            )
            objects.add(picture["object"])
        if verdict not in VERDICTS:
            raise ValueError("请选择判断")
        # Human answers need no citation selection. Regions are optional notes,
        # not a claim of having annotated a minimal sufficient evidence set.
        links = []
        if (
            task == "identity"
            and verdict != "Need more evidence"
            and len(citations) == 2
            and objects == {"subject-A", "subject-B"}
        ):
            links = [
                {
                    "relation": "same_object"
                    if verdict == "Supported"
                    else "different_object",
                    "left": citations[0],
                    "right": citations[1],
                }
            ]
        return citations, links

    def post(self, token, route, body):
        actor = self.actor(token)
        if actor["role"] == "admin":
            raise PermissionError("Admin is read-only")
        if route == "consent":
            if body.get("agree") is not True:
                raise ValueError("请确认自愿参与")
            with self.db:
                self.db.execute(
                    "INSERT OR IGNORE INTO consent VALUES (?,?)",
                    (actor["id"], time.time()),
                )
            return self.state(token)
        if not self.db.execute(
            "SELECT 1 FROM consent WHERE actor=?", (actor["id"],)
        ).fetchone():
            raise ValueError("请先阅读说明")
        index = self.current(actor)
        if index is None:
            raise ValueError("已完成")
        if body.get("index") != index:
            raise ValueError("页面已更新，请刷新当前题")
        task = actor["tasks"][index]
        case = self.case(task)
        self.elapsed(actor, index)
        if route in {"answer", "review"}:
            if (
                not isinstance(body.get("reason"), str)
                or len(body["reason"].strip()) < 2
            ):
                raise ValueError("请简短写出判断理由")
            if (
                type(body.get("confidence")) is not int
                or not 1 <= body["confidence"] <= 5
            ):
                raise ValueError("请选择把握程度")
        env = self.environment(actor, index)
        if route == "action":
            action = body["action"]
            if action.get("type") not in TOOLS - {"finish"}:
                raise ValueError("此操作不可用")
            if (
                action["type"] == "request_photo"
                and task.get("condition") != "interactive"
            ):
                raise ValueError("这组任务只查看已给出的图片")
            result = {}
            error = None
            display_only = action["type"] == "zoom"
            try:
                if display_only:
                    env._image(action["image_id"])
                    factor = action.get("factor")
                    if type(factor) not in (int, float) or not 0.125 <= factor <= 4:
                        raise ActionError("invalid_zoom")
                    result = {"status": "display_zoom", "presentation_only": True}
                else:
                    result = self.human_step(env, action)
            except ActionError as exc:
                error = {"code": exc.code, "details": exc.details}
            event = {
                "action": action,
                "error": error,
                "presentation_only": display_only and error is None,
                "views": [p["view_id"] for p in result.get("images", [])],
                "at": time.time(),
                "result": {k: v for k, v in result.items() if k != "images"},
            }
            self.append(actor, index, event)
            response = self.state(token)
            response["feedback"] = error or event["result"]
            return response
        if task["mode"] == "search":
            if route != "answer":
                raise ValueError("Unknown route")
            citations, links = self.selections(
                self.visible(actor, index),
                body.get("selected", []),
                body["verdict"],
                case["task"],
            )
            action = {
                "type": "finish",
                "verdict": body["verdict"],
                "citations": citations,
                "links": links,
            }
            self.human_step(env, action)
            payload = {
                "status": "finished",
                "task": task,
                "decision": action,
                "confidence": body["confidence"],
                "reason": body["reason"],
                "duration_s": self.elapsed(actor, index),
                "steps": len(self.events(actor, index)),
                "requests": env.requests,
                "tool_cost": env.tool_cost,
                "request_cost": env.request_cost,
                "released": sorted(env._released),
                "events": self.events(actor, index),
            }
        else:
            if route != "review":
                raise ValueError("Unknown route")
            citations, links = self.selections(
                self.visible(actor, index),
                body.get("selected", []),
                body["verdict"],
                case["task"],
            )
            verdict = body["verdict"]
            vote = {
                "protocol": "visual-truth-review-v3",
                "scope": "full_pool_only",
                "reviewer_type": "human",
                "reviewer_id": actor["id"],
                "family_id": task["family_id"],
                "dataset_sha256": self.plan["dataset_sha256"],
                "claim_clear": body.get("clear") is True,
                "reason": body["reason"],
                "verdict": verdict,
                "regions": citations,
                "links": links,
            }
            validate_truth_vote(vote)
            payload = {
                "status": "finished",
                "task": task,
                "vote": vote,
                "events": self.events(actor, index),
                "steps": sum(bool(e.get("action")) for e in self.events(actor, index)),
                "released": sorted(env._released),
                "duration_s": self.elapsed(actor, index),
            }
        if not isinstance(body.get("reason"), str) or len(body["reason"].strip()) < 2:
            raise ValueError("请简短写出判断理由")
        if type(body.get("confidence")) is not int or not 1 <= body["confidence"] <= 5:
            raise ValueError("请选择把握程度")
        payload["ui_protocol"] = UI_PROTOCOL
        payload["citation_scoring_available"] = False
        with self.db:
            self.db.execute(
                "INSERT INTO answers VALUES (?,?,?)",
                (actor["id"], index, json.dumps(payload, ensure_ascii=False)),
            )
        self.environments.pop((actor["id"], index), None)
        return {"submitted": True, "message": "已保存。感谢你的独立判断。"}

    def admin(self):
        counts = Counter(r[0] for r in self.db.execute("SELECT actor FROM answers"))
        preview_access = self.root / "preview" / "access.json"
        preview_token = (
            next(reversed(json.loads(preview_access.read_text())))
            if preview_access.exists()
            else None
        )
        return {
            "preview_token": preview_token,
            "role": "admin",
            "ui_protocol": UI_PROTOCOL,
            "limits": {"time": None, "steps": None, "points": None},
            "plan": {
                k: v
                for k, v in self.plan.items()
                if k
                not in {
                    "data",
                    "family_ids",
                    "budget",
                    "max_turns",
                    "search_seconds",
                    "review_seconds",
                }
            },
            "invitations": [
                {
                    "id": a["id"],
                    "token": token,
                    "completed": counts[a["id"]],
                    "total": len(a["tasks"]),
                }
                for token, a in self.access.items()
                if a["role"] == "participant"
            ],
        }

    def export(self, votes_only=False):
        rows = [
            {"actor": a, "index": i, **json.loads(p)}
            for a, i, p in self.db.execute(
                "SELECT actor,idx,payload FROM answers ORDER BY actor,idx"
            )
        ]
        return [r["vote"] for r in rows if r.get("vote")] if votes_only else rows


def create_server(root, port=8765):
    main_study = Study(root)
    studies = [main_study]
    if (Path(root) / "preview" / "plan.json").exists():
        studies.append(Study(Path(root) / "preview"))

    def for_token(token):
        for study in studies:
            if token in study.access:
                return study
        raise PermissionError("Invalid invitation")

    web = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def token(self):
            value = self.headers.get("Authorization", "")
            return value[7:] if value.startswith("Bearer ") else ""

        def respond(self, content, mime="application/json", status=200):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path in ("/", "/app.js", "/i18n.js", "/style.css"):
                name = {
                    "/": "index.html",
                    "/app.js": "app.js",
                    "/i18n.js": "i18n.js",
                    "/style.css": "style.css",
                }[self.path]
                return self.respond(
                    (web / name).read_bytes(),
                    "text/html; charset=utf-8"
                    if name.endswith("html")
                    else "text/javascript; charset=utf-8"
                    if name.endswith("js")
                    else "text/css; charset=utf-8",
                )
            if self.path == "/demo.webp":
                path = (
                    Path(__file__).resolve().parents[2]
                    / "docs/demo/assets/operations.webp"
                )
                return self.respond(path.read_bytes(), "image/webp")
            try:
                study = for_token(self.token())
                with study.lock:
                    actor = study.actor(self.token())
                    if self.path == "/api/state":
                        state = study.state(self.token())
                        state["preview"] = study.root.name == "preview"
                        return self.respond(state)
                    if self.path in ("/api/export", "/api/votes"):
                        if actor["role"] != "admin":
                            raise PermissionError("Private organizer route")
                        return self.respond(study.export(self.path.endswith("votes")))
                    if self.path.startswith("/api/image/"):
                        if actor["role"] == "admin":
                            raise PermissionError("No assigned image")
                        index = study.current(actor)
                        if index is None:
                            raise PermissionError("No active task")
                        if not study.db.execute(
                            "SELECT 1 FROM consent WHERE actor=?", (actor["id"],)
                        ).fetchone():
                            raise PermissionError("Consent required")
                        ref = unquote(self.path[len("/api/image/") :])
                        allowed = {p["view_id"] for p in study.visible(actor, index)}
                        if ref not in allowed:
                            raise PermissionError("Image is not released in this stage")
                        picture = study.pictures.get((actor["id"], index, ref))
                        if not picture:
                            picture = next(
                                p
                                for p in study.visible(actor, index)
                                if p["view_id"] == ref
                            )
                        return self.respond(
                            picture["bytes"],
                            "image/png"
                            if picture["bytes"][:8] == b"\x89PNG\r\n\x1a\n"
                            else "image/jpeg",
                        )
                self.respond({"error": "Not found"}, status=404)
            except PermissionError:
                self.respond({"error": "邀请链接无效或无权访问"}, status=403)
            except (ValueError, KeyError, OSError, ActionError):
                self.respond({"error": "请求无效，请刷新当前题"}, status=400)

        def do_POST(self):
            try:
                if not self.path.startswith("/api/"):
                    raise ValueError("Unknown route")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected JSON")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 50000:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                if (
                    self.path in {"/api/action", "/api/answer", "/api/review"}
                    and body.get("ui_protocol") != UI_PROTOCOL
                ):
                    raise ValueError("页面已更新，请刷新当前题")
                study = for_token(self.token())
                with study.lock:
                    result = study.post(self.token(), self.path[len("/api/") :], body)
                result["preview"] = study.root.name == "preview"
                self.respond(result)
            except PermissionError:
                self.respond({"error": "邀请链接无效或无权访问"}, status=403)
            except (ValueError, KeyError, TypeError, ActionError) as exc:
                self.respond({"error": str(exc)}, status=400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(root, port=8765):
    server = create_server(root, port)
    print(
        f"Short study listening on 127.0.0.1:{port}; responses persist in SQLite",
        flush=True,
    )
    server.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    serve(args.study, args.port)


if __name__ == "__main__":
    main()
