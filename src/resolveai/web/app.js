"use strict";
let token = decodeURIComponent(location.hash.slice(1)),
  state = null,
  boxes = {},
  corners = {},
  marked = new Set(),
  selected = new Set(),
  urls = [],
  timer = null,
  busy = false,
  endAt = 0,
  invitations = [];
const $ = (id) => document.getElementById(id);
async function api(path, body) {
  const r = await fetch("/api/" + path, {
    method: body ? "POST" : "GET",
    headers: {
      Authorization: "Bearer " + token,
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json();
  if (!r.ok) throw Error(data.error || "请求失败");
  return data;
}
function section(id) {
  ["welcome", "intro", "task", "saved", "done", "admin"].forEach(
    (name) => ($(name).hidden = name !== id),
  );
  $("error").textContent = "";
}
function message(error) {
  $("error").textContent = error.message || String(error);
}
async function guarded(fn) {
  if (busy) return;
  busy = true;
  document.querySelectorAll("button").forEach((b) => (b.disabled = true));
  try {
    await fn();
  } catch (e) {
    message(e);
  } finally {
    busy = false;
    document.querySelectorAll("button").forEach((b) => (b.disabled = false));
  }
}
function options(id, values, labels) {
  $(id).replaceChildren(
    ...values.map((v, i) => {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = labels ? labels[i] : v;
      return o;
    }),
  );
}
function actorText(p) {
  return p.object === "normal-reference"
    ? "正常参考（另一个实物）"
    : p.object === "subject-A"
      ? "提交对象 A"
      : p.object === "subject-B"
        ? "提交对象 B"
        : "目标图片";
}
function evidence() {
  return [...selected].map((ref) => ({ view_id: ref, bbox: boxes[ref] }));
}
function verdict() {
  return document.querySelector("input[name=verdict]:checked")?.value;
}
function resetAnswer() {
  document
    .querySelectorAll("input[name=verdict]")
    .forEach((r) => (r.checked = false));
  $("reason").value = "";
  $("confidence").value = "3";
  selected.clear();
}
function stopClock() {
  clearInterval(timer);
  $("clock").textContent = "";
}
function startClock(seconds) {
  clearInterval(timer);
  endAt = Date.now() + seconds * 1000;
  timer = setInterval(() => {
    const n = Math.max(0, Math.ceil((endAt - Date.now()) / 1000));
    $("clock").textContent = "本题剩余 " + n + " 秒";
    $("clock").className = n < 16 ? "warning" : "";
    if (n === 0 && !busy) {
      clearInterval(timer);
      guarded(async () => {
        await show(await api("timeout", { index: state.index }));
      });
    }
  }, 500);
}
async function picture(p) {
  const card = document.createElement("div");
  card.className = "card";
  const label = document.createElement("label"),
    check = document.createElement("input");
  check.type = "checkbox";
  check.dataset.ref = p.view_id;
  check.checked = selected.has(p.view_id);
  check.onchange = () =>
    check.checked ? selected.add(p.view_id) : selected.delete(p.view_id);
  label.append(
    check,
    document.createTextNode(
      " " +
        actorText(p) +
        " · " +
        (p.camera_view === "preview" || p.camera_view === "overview"
          ? "初始小图"
          : p.camera_view === "reference"
            ? "正常参考"
            : p.camera_view === "original"
              ? "原图"
              : p.camera_view || ""),
    ),
  );
  card.append(label);
  const wrap = document.createElement("div");
  wrap.className = "picture";
  const img = document.createElement("img"),
    canvas = document.createElement("canvas");
  img.alt = actorText(p);
  boxes[p.view_id] ??= p.target_bbox || p.source_bbox;
  const response = await fetch("/api/image/" + encodeURIComponent(p.view_id), {
    headers: { Authorization: "Bearer " + token },
  });
  if (!response.ok) throw Error("图片加载失败，请刷新");
  const url = URL.createObjectURL(await response.blob());
  urls.push(url);
  img.src = url;
  const note = document.createElement("small");
  note.textContent =
    state.task === "identity" && !p.target_bbox
      ? "补充视角：请点击两角标出对应目标"
      : "点击两角标记证据；默认绿色框为目标区域";
  function draw() {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d"),
      b = p.source_bbox,
      a = boxes[p.view_id];
    if (state.task === "identity" && !p.target_bbox && !marked.has(p.view_id))
      return;
    ctx.strokeStyle = "#14a06f";
    ctx.lineWidth = Math.max(2, canvas.width / 160);
    ctx.strokeRect(
      ((a[0] - b[0]) / (b[2] - b[0])) * canvas.width,
      ((a[1] - b[1]) / (b[3] - b[1])) * canvas.height,
      ((a[2] - a[0]) / (b[2] - b[0])) * canvas.width,
      ((a[3] - a[1]) / (b[3] - b[1])) * canvas.height,
    );
  }
  img.onload = draw;
  img.onclick = (e) => {
    const r = img.getBoundingClientRect(),
      b = p.source_bbox,
      x = Math.max(
        b[0],
        Math.min(
          b[2],
          Math.round(b[0] + ((e.clientX - r.left) / r.width) * (b[2] - b[0])),
        ),
      ),
      y = Math.max(
        b[1],
        Math.min(
          b[3],
          Math.round(b[1] + ((e.clientY - r.top) / r.height) * (b[3] - b[1])),
        ),
      );
    if (!corners[p.view_id]) {
      corners[p.view_id] = [x, y];
      note.textContent = "再点击区域对角";
    } else {
      const [a, c] = corners[p.view_id];
      delete corners[p.view_id];
      if (a === x || c === y) {
        note.textContent = "区域太小，请重新点两角";
        return;
      }
      boxes[p.view_id] = [
        Math.min(a, x),
        Math.min(c, y),
        Math.max(a, x),
        Math.max(c, y),
      ];
      note.textContent = "区域已标记";
      marked.add(p.view_id);
      draw();
      check.checked = true;
      selected.add(p.view_id);
    }
  };
  wrap.append(img, canvas);
  card.append(wrap, note);
  return card;
}
async function show(data) {
  state = data;
  stopClock();
  if (data.role === "admin") {
    section("admin");
    invitations = data.invitations;
    $("study-info").textContent =
      "20 人，每人 3 道搜证 + 2 道独立审核。记录：" +
      data.invitations.reduce((n, p) => n + p.completed, 0) +
      " / 100。";
    $("invitations").replaceChildren(
      ...data.invitations.map((p) => {
        const tr = document.createElement("tr");
        [p.id, p.completed + " / " + p.total].forEach((s) => {
          const td = document.createElement("td");
          td.textContent = s;
          tr.append(td);
        });
        const td = document.createElement("td"),
          a = document.createElement("a");
        a.href = location.origin + "/#" + p.token;
        a.textContent = "打开 " + p.id + " 的邀请链接";
        a.target = "_blank";
        a.rel = "noreferrer";
        td.append(a);
        tr.append(td);
        return tr;
      }),
    );
    return;
  }
  if (data.intro) {
    section("intro");
    return;
  }
  if (data.done) {
    section("done");
    return;
  }
  if (data.submitted) {
    section("saved");
    $("saved-message").textContent = data.message;
    return;
  }
  section("task");
  $("progress").textContent =
    "第 " +
    (data.index + 1) +
    " / " +
    data.total +
    " 题 · " +
    (data.mode === "review" ? "证据审核" : "搜证任务");
  $("claim").textContent = data.claim;
  $("context").textContent = data.context;
  $("original").textContent = data.original_claim;
  $("tools").hidden = data.mode === "review";
  $("review-checks").hidden = data.mode !== "review" || data.phase !== 4;
  $("condition").textContent =
    data.mode === "review"
      ? data.phase_title + "（仅依据当前显示图片；前面的判断已锁定）"
      : {
          initial: "本题只提供初始图片，可检查、裁剪、放大，但不能索取新照片。",
          interactive: "可以操作已有图片，也可以请求照片。",
          full_available: "本题已提供所有可获取照片；没有额外照片可请求。",
        }[data.condition] +
        " 剩余预算：" +
        data.budget +
        "，已操作：" +
        data.turns +
        " / " +
        (data.max_turns - 1);
  $("submit").textContent =
    data.mode === "review" && data.phase < 4
      ? "保存，查看下一组材料"
      : "提交本题";
  const old = urls;
  urls = [];
  $("pictures").replaceChildren();
  const cards = await Promise.all(data.images.map(picture));
  $("pictures").replaceChildren(...cards);
  old.forEach(URL.revokeObjectURL);
  startClock(data.remaining_seconds);
  if (data.mode === "search") {
    options(
      "image",
      data.images.map((p) => p.view_id),
      data.images.map((p, i) => actorText(p) + " · 图 " + (i + 1)),
    );
    options(
      "object",
      data.request_options.objects || [],
      (data.request_options.objects || []).map((o) =>
        o === "subject-A"
          ? "对象 A"
          : o === "subject-B"
            ? "对象 B"
            : "目标物品",
      ),
    );
    options(
      "time",
      data.request_options.times || [],
      (data.request_options.times || []).map((t) =>
        t === "capture" ? "当前拍摄时刻" : t,
      ),
    );
    options(
      "view",
      data.request_options.views || [],
      (data.request_options.views || []).map((v) =>
        v === "original"
          ? "原图"
          : v === "overview"
            ? "概览照片"
            : v.startsWith("view-")
              ? "视角 " + Number(v.slice(5))
              : v,
      ),
    );
    if ((data.request_options.views || []).length > 1) {
      const o = document.createElement("option");
      o.value = "";
      o.textContent = "选择照片视角";
      o.selected = true;
      $("view").prepend(o);
    }
    if ((data.request_options.objects || []).length > 1) {
      const o = document.createElement("option");
      o.value = "";
      o.textContent = "选择对象";
      o.selected = true;
      $("object").prepend(o);
    }
    $("operation").querySelector("option[value=request_photo]").disabled =
      data.condition !== "interactive";
    $("operation").querySelector("option[value=ocr]").disabled =
      !data.ocr_available;
    toolVisibility();
  }
  if (data.feedback) {
    const f = data.feedback;
    $("feedback").textContent = f.code
      ? "操作未成功：" +
        f.code +
        (f.details?.suggestion ? "；先裁剪再放大" : "")
      : f.status === "unable_to_provide"
        ? "提供方无法提供符合请求的照片；这不能说明声明是真是假。"
        : f.text_regions
          ? "识别文字：" + f.text_regions.map((r) => r.text).join(" / ")
          : "操作已完成。";
  }
}
function toolVisibility() {
  const op = $("operation").value;
  $("image").hidden = op === "request_photo" || op === "compare";
  $("factor").hidden = op !== "zoom";
  $("requests").hidden = op !== "request_photo";
}
$("operation").onchange = toolVisibility;
$("begin").onclick = () =>
  guarded(async () => {
    if (!$("agree").checked) throw Error("请先确认理解并自愿参与");
    await show(await api("consent", { agree: true }));
  });
$("operate").onclick = () =>
  guarded(async () => {
    const type = $("operation").value,
      ref = $("image").value,
      p = state.images.find((p) => p.view_id === ref);
    let action = { type };
    if (type === "request_photo") {
      if (!$("view").value || !$("object").value)
        throw Error("请选择要请求的对象与照片视角");
      action.query = {
        object: $("object").value,
        time: $("time").value,
        view: $("view").value,
      };
    } else if (type === "compare") {
      const refs = [...selected];
      if (refs.length !== 2) throw Error("请勾选两张要比较的图片");
      action.image_ids = refs;
    } else {
      action.image_id = ref;
      if (type === "zoom") action.factor = Number($("factor").value);
      if (type === "crop") {
        const b = boxes[ref],
          s = p.source_bbox,
          w = p.display_size[0],
          h = p.display_size[1];
        action.bbox = [
          Math.max(0, Math.floor(((b[0] - s[0]) / (s[2] - s[0])) * w)),
          Math.max(0, Math.floor(((b[1] - s[1]) / (s[3] - s[1])) * h)),
          Math.min(w, Math.ceil(((b[2] - s[0]) / (s[2] - s[0])) * w)),
          Math.min(h, Math.ceil(((b[3] - s[1]) / (s[3] - s[1])) * h)),
        ];
      }
    }
    await show(await api("action", { index: state.index, action }));
  });
$("submit").onclick = () =>
  guarded(async () => {
    const v = verdict();
    if (!v) throw Error("请选择判断");
    if ($("reason").value.trim().length < 2) throw Error("请简短写出理由");
    const payload = {
      index: state.index,
      verdict: v,
      reason: $("reason").value,
      confidence: Number($("confidence").value),
      selected: evidence(),
      phase: state.phase,
      clear: $("clear").checked,
      checked: $("checked").checked,
    };
    const response = await api(
      state.mode === "review" ? "review" : "answer",
      payload,
    );
    resetAnswer();
    await show(response);
  });
$("next").onclick = () =>
  guarded(async () => {
    boxes = {};
    corners = {};
    marked.clear();
    resetAnswer();
    await show(await api("state"));
  });
$("refresh").onclick = () => guarded(async () => show(await api("state")));
async function download(route, name, jsonl = false) {
  const data = await api(route);
  const blob = new Blob(
      [
        jsonl
          ? data.map((r) => JSON.stringify(r)).join("\n") + "\n"
          : JSON.stringify(data, null, 2),
      ],
      { type: "application/json" },
    ),
    a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$("export").onclick = () =>
  guarded(() => download("export", "resolveai-human-responses.json"));
$("votes").onclick = () =>
  guarded(() => download("votes", "resolveai-independent-votes.jsonl", true));
$("copy-links").onclick = () =>
  guarded(async () => {
    await navigator.clipboard.writeText(
      invitations
        .map((p) => p.id + " " + location.origin + "/#" + p.token)
        .join("\n"),
    );
    $("study-info").textContent = "20 个个人邀请链接已复制。";
  });
if (token) guarded(async () => show(await api("state")));
else section("welcome");

window.addEventListener("hashchange", () => location.reload());
