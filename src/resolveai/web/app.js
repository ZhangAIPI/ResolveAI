"use strict";
let token = decodeURIComponent(location.hash.slice(1)),
  state = null,
  boxes = {},
  corners = {},
  marked = new Set(),
  imageCache = new Map(),
  busy = false,
  invitations = [];
const $ = (id) => document.getElementById(id);
async function api(path, body) {
  const r = await fetch("/api/" + path, {
    method: body ? "POST" : "GET",
    headers: {
      Authorization: "Bearer " + token,
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    body: body
      ? JSON.stringify({ ...body, ui_protocol: "human-ui-v5-optional-regions" })
      : undefined,
  });
  const data = await r.json();
  if (!r.ok) throw Error(localizedError(data.error) || t("requestFailed"));
  return data;
}
function section(id) {
  ["welcome", "intro", "task", "saved", "done", "admin"].forEach(
    (name) => ($(name).hidden = name !== id),
  );
  $("error").textContent = "";
  if (id !== "task") $("step-count").textContent = "";
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
function clippedBox(p, box) {
  const b = p.source_bbox,
    a = box || b;
  const result = [
    Math.max(a[0], b[0]),
    Math.max(a[1], b[1]),
    Math.min(a[2], b[2]),
    Math.min(a[3], b[3]),
  ];
  return result[0] < result[2] && result[1] < result[3] ? result : [...b];
}
function displayImages(data) {
  const refs = new Set(data.display_views || data.images.map((p) => p.view_id));
  const byRef = new Map(data.images.map((p) => [p.view_id, p]));
  const shown = [...refs].map((ref) => byRef.get(ref)).filter(Boolean);
  for (const ref of [...marked]) {
    if (refs.has(ref)) continue;
    const old = data.images.find((p) => p.view_id === ref);
    const replacement = shown.find((p) => p.source_id === old?.source_id);
    if (!replacement) continue;
    boxes[replacement.view_id] = clippedBox(replacement, boxes[ref]);
    marked.delete(ref);
    marked.add(replacement.view_id);
  }
  return shown;
}
function actorText(p) {
  return p.object === "normal-reference"
    ? t("reference")
    : p.object === "subject-A"
      ? t("subjectA")
      : p.object === "subject-B"
        ? t("subjectB")
        : t("target");
}
function evidence() {
  return [...marked].map((ref) => ({ view_id: ref, bbox: boxes[ref] }));
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
  marked.clear();
}
function refreshRegions() {
  document.querySelectorAll(".card").forEach((card) => {
    card.querySelector(".clear-mark").hidden = !marked.has(card.dataset.ref);
    card.querySelector(".selection-status").textContent = marked.has(
      card.dataset.ref,
    )
      ? t("regionDone")
      : "";
  });
  $("selection-hint").textContent = t("optionalRegions");
}
function displayBox(p, box) {
  const b = p.source_bbox,
    [w, h] = p.display_size;
  return [
    Math.max(0, Math.floor(((box[0] - b[0]) / (b[2] - b[0])) * w)),
    Math.max(0, Math.floor(((box[1] - b[1]) / (b[3] - b[1])) * h)),
    Math.min(w, Math.ceil(((box[2] - b[0]) / (b[2] - b[0])) * w)),
    Math.min(h, Math.ceil(((box[3] - b[1]) / (b[3] - b[1])) * h)),
  ];
}
function perform(action) {
  return guarded(async () => {
    await show(await api("action", { index: state.index, action }));
  });
}
async function picture(p) {
  const card = document.createElement("div");
  card.className = "card";
  card.dataset.ref = p.view_id;
  const title = document.createElement("div");
  title.className = "picture-title";
  const view =
    p.camera_view === "preview" || p.camera_view === "overview"
      ? t("small")
      : p.camera_view === "reference"
        ? t("normal")
        : p.camera_view === "original"
          ? t("raw")
          : p.camera_view || "";
  title.textContent =
    actorText(p) + (p.object === "normal-reference" ? "" : " · " + view);
  card.append(title);
  if (
    Math.max(...p.display_size) < 128 &&
    ["preview", "overview"].includes(p.camera_view)
  ) {
    const quality = document.createElement("small");
    quality.className = "photo-quality";
    quality.textContent = t("lowResolution", p.display_size.join(" × "));
    card.append(quality);
  }
  const wrap = document.createElement("div");
  wrap.className = "picture";
  const baseWidth = Math.min(
    500,
    (360 * p.display_size[0]) / p.display_size[1],
  );
  wrap.style.setProperty("--picture-width", baseWidth + "px");
  wrap.style.setProperty(
    "--content-width",
    baseWidth * (p.display_scale || 1) + "px",
  );
  const choose = document.createElement("button");
  choose.type = "button";
  choose.className = "picture-surface";
  choose.setAttribute("aria-label", title.textContent);
  const img = document.createElement("img"),
    canvas = document.createElement("canvas");
  img.alt = actorText(p);
  boxes[p.view_id] ??= clippedBox(p, p.target_bbox);
  let url = imageCache.get(p.view_id);
  if (!url) {
    const response = await fetch(
      "/api/image/" + encodeURIComponent(p.view_id),
      {
        headers: { Authorization: "Bearer " + token },
      },
    );
    if (!response.ok) throw Error(t("imageFailed"));
    url = URL.createObjectURL(await response.blob());
    imageCache.set(p.view_id, url);
  }
  img.src = url;
  const badge = document.createElement("span");
  badge.className = "selection-status";
  const note = document.createElement("small");
  note.className = "region-note";
  let regionMode = null;
  function draw() {
    // Cached images can finish loading before their card enters the document.
    // Never stretch a 1px canvas over a photo; redraw when layout has a size.
    if (!img.clientWidth || !img.clientHeight) return;
    canvas.width = Math.round(img.clientWidth);
    canvas.height = Math.round(img.clientHeight);
    const ctx = canvas.getContext("2d"),
      b = p.source_bbox;
    function outline(box, color, dashed) {
      const a = [
        Math.max(box[0], b[0]), Math.max(box[1], b[1]),
        Math.min(box[2], b[2]), Math.min(box[3], b[3]),
      ];
      if (a[0] >= a[2] || a[1] >= a[3]) return;
      const x = ((a[0] - b[0]) / (b[2] - b[0])) * canvas.width;
      const y = ((a[1] - b[1]) / (b[3] - b[1])) * canvas.height;
      const w = ((a[2] - a[0]) / (b[2] - b[0])) * canvas.width;
      const h = ((a[3] - a[1]) / (b[3] - b[1])) * canvas.height;
      ctx.setLineDash(dashed ? [8, 4] : []);
      ctx.strokeStyle = "white";
      ctx.lineWidth = 5;
      ctx.strokeRect(x, y, w, h);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);
    }
    if (p.target_bbox) outline(p.target_bbox, "#d35400", true);
    if (marked.has(p.view_id)) outline(boxes[p.view_id], "#2563eb", false);
  }
  const observer = new ResizeObserver(draw);
  observer.observe(img);
  card.stopObserving = () => observer.disconnect();
  img.onload = draw;
  choose.onclick = (event) => {
    if (!regionMode) return;
    if (!event.detail) return;
    const r = img.getBoundingClientRect(),
      b = p.source_bbox;
    const x = Math.max(
      b[0],
      Math.min(
        b[2],
        Math.round(b[0] + ((event.clientX - r.left) / r.width) * (b[2] - b[0])),
      ),
    );
    const y = Math.max(
      b[1],
      Math.min(
        b[3],
        Math.round(b[1] + ((event.clientY - r.top) / r.height) * (b[3] - b[1])),
      ),
    );
    if (!corners[p.view_id]) {
      corners[p.view_id] = [x, y];
      note.textContent = t("corner");
      return;
    }
    const [a, c] = corners[p.view_id];
    delete corners[p.view_id];
    if (a === x || c === y) {
      note.textContent = t("smallRegion");
      return;
    }
    const mode = regionMode;
    regionMode = null;
    choose.classList.remove("marking");
    const regionBox = [
      Math.min(a, x),
      Math.min(c, y),
      Math.max(a, x),
      Math.max(c, y),
    ];
    if (mode === "mark") {
      boxes[p.view_id] = regionBox;
      marked.add(p.view_id);
    }
    note.textContent = t("regionDone");
    draw();
    refreshRegions();
    if (mode === "crop")
      perform({
        type: "crop",
        image_id: p.view_id,
        bbox: displayBox(p, regionBox),
      });
  };
  function region(mode) {
    if (regionMode === mode) {
      regionMode = null;
      delete corners[p.view_id];
      choose.classList.remove("marking");
      note.textContent = "";
      return;
    }
    regionMode = mode;
    delete corners[p.view_id];
    choose.classList.add("marking");
    note.textContent = mode === "crop" ? t("cropHint") : t("markHint");
  }
  function button(label, handler) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.onclick = handler;
    return b;
  }
  choose.append(img, canvas);
  wrap.append(choose);
  card.append(wrap, badge);
  const controls = document.createElement("div");
  controls.className = "picture-controls";
  {
    controls.append(
      button(t("zoomIn"), () =>
        perform({ type: "zoom", image_id: p.view_id, factor: 2 }),
      ),
      button(t("zoomOut"), () =>
        perform({ type: "zoom", image_id: p.view_id, factor: 0.5 }),
      ),
      button(t("crop"), () => region("crop")),
      button(t("mark"), () => region("mark")),
    );
  }
  const clearMark = button(t("clearMark"), () => {
    marked.delete(p.view_id);
    note.textContent = "";
    draw();
    refreshRegions();
  });
  clearMark.className = "clear-mark";
  clearMark.hidden = !marked.has(p.view_id);
  controls.append(clearMark);
  if (p.view_id !== p.image_id) {
    controls.append(
      button(t("restore"), () =>
        perform({ type: "inspect", image_id: p.image_id }),
      ),
    );
  }
  if (state.ocr_available) {
    controls.append(
      button(t("ocr"), () => perform({ type: "ocr", image_id: p.view_id })),
    );
  }
  card.append(controls, note);
  return card;
}
function photoRequests(data) {
  $("request-panel").hidden = data.condition !== "interactive";
  $("request-buttons").replaceChildren();
  if (data.condition !== "interactive") return;
  const o = data.request_options;
  for (const object of o.objects || []) {
    for (const time of o.times || []) {
      for (const view of o.views || []) {
        const b = document.createElement("button");
        b.type = "button";
        const subject =
          object === "subject-A"
            ? "A"
            : object === "subject-B"
              ? "B"
              : t("targetRequest");
        const name =
          view === "original"
            ? t("raw")
            : view === "overview"
              ? t("overview")
              : view.startsWith("view-")
                ? t("view", Number(view.slice(5)))
                : view;
        b.textContent =
          subject +
          " · " +
          name +
          ((o.times || []).length > 1 ? " · " + time : "");
        b.onclick = () =>
          perform({ type: "request_photo", query: { object, time, view } });
        $("request-buttons").append(b);
      }
    }
  }
}
async function show(data) {
  state = data;
  $("preview-notice").hidden = !data.preview;
  if (data.role === "admin") {
    section("admin");
    invitations = data.invitations;
    $("preview-link").hidden = !data.preview_token;
    if (data.preview_token)
      $("preview-link").href = location.origin + "/#" + data.preview_token;
    $("study-info").textContent = t(
      "organizerInfo",
      data.invitations.reduce((n, p) => n + p.completed, 0),
    );
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
        a.textContent = t("invitation", p.id);
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
    $("saved-message").textContent = t("saved");
    return;
  }
  section("task");
  $("progress").textContent = t(
    "progress",
    data.index + 1,
    data.total,
    t(data.mode === "review" ? "review" : "search"),
  );
  $("claim").textContent = language === "en" ? data.original_claim : data.claim;
  $("context").textContent = t(
    data.task === "identity" ? "identityContext" : "stateContext",
  );
  $("original").textContent = data.original_claim;
  $("tools").hidden = false;
  $("step-count").textContent = t("steps", data.turns);
  $("supported-label").textContent =
    data.task === "identity" ? t("same") : t("supported");
  $("refuted-label").textContent =
    data.task === "identity" ? t("different") : t("refuted");
  $("review-checks").hidden = data.mode !== "review";
  $("condition").textContent =
    data.mode === "review"
      ? t("singleReview")
      : t(
          {
            initial: "initial",
            interactive: "interactive",
            full_available: "full",
          }[data.condition],
        );
  $("submit").textContent = t("submit");
  $("pictures")
    .querySelectorAll(".card")
    .forEach((c) => c.stopObserving?.());
  $("pictures").replaceChildren();
  const shown = displayImages(data);
  const cards = await Promise.all(shown.map(picture));
  $("pictures").replaceChildren(...cards);
  const keep = new Set(shown.flatMap((p) => [p.view_id, p.image_id]));
  for (const [ref, url] of imageCache) {
    if (!keep.has(ref)) {
      URL.revokeObjectURL(url);
      imageCache.delete(ref);
    }
  }
  refreshRegions();
  photoRequests(data);
  $("feedback").textContent = "";
  if (data.feedback) {
    const f = data.feedback;
    $("feedback").textContent = f.code
      ? f.code === "view_too_large"
        ? t("tooLarge")
        : t("actionError", f.code)
      : f.status === "unable_to_provide"
        ? t("unavailable")
        : f.text_regions
          ? t("textResult", f.text_regions.map((r) => r.text).join(" / "))
          : t("actionDone");
  }
}
$("begin").onclick = () =>
  guarded(async () => {
    if (!$("agree").checked) throw Error(t("agreeError"));
    await show(await api("consent", { agree: true }));
  });
$("submit").onclick = () =>
  guarded(async () => {
    const v = verdict();
    if (!v) throw Error(t("verdictError"));
    if ($("reason").value.trim().length < 2) throw Error(t("reasonError"));
    const payload = {
      index: state.index,
      verdict: v,
      reason: $("reason").value,
      confidence: Number($("confidence").value),
      selected: evidence(),
      clear: $("clear").checked,
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
    imageCache.forEach(URL.revokeObjectURL);
    imageCache.clear();
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
    $("study-info").textContent = t("copied");
  });
$("language").onclick = () =>
  guarded(async () => {
    language = language === "zh" ? "en" : "zh";
    localStorage.setItem("resolveai-language", language);
    applyLanguage();
    if (state?.images || state?.role === "admin") await show(state);
  });
applyLanguage();
if (token) guarded(async () => show(await api("state")));
else section("welcome");

window.addEventListener("hashchange", () => location.reload());
