"use strict";
let language =
  localStorage.getItem("resolveai-language") === "en" ? "en" : "zh";
const UI_TEXT = {
  lowResolution: [
    "低清照片（{0} 像素）；已放大显示",
    "Low-resolution photo ({0} px), displayed enlarged",
  ],
  singleReview: [
    "全部可获取照片已显示。可放大或裁剪；这道题只提交一次。",
    "All available photos are shown. You can zoom or crop; submit this task once.",
  ],
  previewLink: [
    "试用新版流程（不计入统计）",
    "Try the updated flow (excluded from study results)",
  ],
  previewNotice: [
    "试用模式：回答不计入实验统计。",
    "Preview mode: responses are excluded from study results.",
  ],
  title: ["ResolveAI · 看图与搜证", "ResolveAI · Visual evidence study"],
  header: ["看图与搜证 · 约 5–10 分钟", "Visual evidence · about 5–10 minutes"],
  welcomeEyebrow: ["人工验证实验", "Human study"],
  welcomeTitle: [
    "这些照片，足以判断吗？",
    "Are these photos enough to decide?",
  ],
  welcomePurpose: [
    "我们想知道：题目是否清楚、照片是否能帮助判断，以及多拿一张照片能否改变结论。",
    "We want to learn whether the question is clear, the photos support a decision, and another photo changes that decision.",
  ],
  welcomeTasks: [
    "每位参与者完成 3 道搜证题与 2 道审核题。看不清时可以选择“无法确定”。",
    "You will complete 3 evidence-search tasks and 2 photo reviews. You can choose “Cannot determine” when the evidence is unclear.",
  ],
  inviteInfo: [
    "请使用电脑或平板，并用研究者发给你的个人邀请链接进入。不要转发链接，也不要与其他参与者讨论题目。",
    "Use a computer or tablet and your personal invitation link. Do not share the link or discuss the tasks with others.",
  ],
  demo: ["看看操作示例", "See an example"],
  demoNote: [
    "示例来自可控渲染，不是正式实验题。",
    "The example uses rendered images and is not a study task.",
  ],
  introTitle: ["开始前，花半分钟看说明", "Before you begin"],
  introCount: [
    "共 5 道短任务，预计 5–10 分钟。没有倒计时、点数或操作次数限制。",
    "There are 5 short tasks, usually taking 5–10 minutes. There is no timer, points budget or action limit.",
  ],
  introFact: [
    "只判断图片里的事实。正常参考来自另一个实物，不是目标的“之前照片”。",
    "Judge only what the photos show. A normal reference is another object, not a “before” photo of the target.",
  ],
  introClick: [
    "直接选择判断并提交。不必选图片；想指出细节时，点“画框（可选）”。",
    "Choose your answer and submit. No photo selection is needed; use “Mark a region (optional)” to point out a detail.",
  ],
  introZoom: [
    "放大只改变显示；请求照片才可能带来尚未看到的材料。",
    "Zoom changes the display; requesting photos may provide evidence you have not seen.",
  ],
  introReview: [
    "每道题只提交一次，提交后进入另一道题。请独立作答。",
    "Submit each task once, then continue to a different task. Work independently.",
  ],
  consent: [
    "我理解以上说明并自愿参与。记录包括答案、选图区域、操作与用时；可随时关闭网页退出。",
    "I understand and agree to participate. We record answers, evidence regions, actions and time spent. I can leave by closing the page.",
  ],
  begin: ["开始", "Begin"],
  original: ["原始题目", "Original wording"],
  answer: ["你的判断", "Your decision"],
  optionalRegions: [
    "直接选择判断并提交即可。画框可选，用来指出你关注的细节。",
    "Choose your answer and submit. Optionally mark a region to point out a detail.",
  ],
  requestPanel: ["需要其他照片？", "Need other photos?"],
  optional: ["可选：判断把握", "Optional: confidence"],
  confidence: ["把握程度", "Confidence"],
  reason: ["一句话理由", "A short reason"],
  reasonPlaceholder: [
    "例如：瓶口左边可见缺口；或缺少能确认身份的细节",
    "For example: a chip is visible on the left rim, or identity details are missing.",
  ],
  clear: ["声明和目标清楚", "The claim and target are clear"],
  checked: [
    "已经检查其他视角和小图能否提供替代证据",
    "I checked whether other views or previews provide alternative evidence",
  ],
  next: ["继续下一题", "Next task"],
  done: ["谢谢，实验已完成。", "Thank you. You have finished."],
  doneNote: [
    "你的回答已保存。这里不显示答案或模型结果。",
    "Your responses are saved. Answers and model results are not shown here.",
  ],
  admin: ["研究者入口", "Organizer dashboard"],
  adminNote: [
    "将不同邀请链接分别发给 20 位参与者，每人使用一个链接。审核题已自动分配。",
    "Send a different invitation link to each of 20 participants. Review tasks are already assigned.",
  ],
  refresh: ["刷新进度", "Refresh"],
  export: ["下载匿名记录", "Download anonymous responses"],
  votes: ["下载独立审核 JSONL", "Download review JSONL"],
  copyLinks: ["复制全部邀请链接", "Copy all invitations"],
  id: ["编号", "ID"],
  complete: ["完成", "Completed"],
  personal: ["个人邀请链接", "Personal invitation"],
  footer: [
    "ResolveAI · 仅验证视觉事实 · 相似不等于同一实物",
    "ResolveAI · Visual facts only · Similarity does not establish identity",
  ],
  low1: ["很低", "Very low"],
  low2: ["较低", "Low"],
  mid: ["一般", "Moderate"],
  high4: ["较高", "High"],
  high5: ["很高", "Very high"],
  reference: ["正常参考（另一个实物）", "Normal reference (another object)"],
  subjectA: ["对象 A", "Object A"],
  subjectB: ["对象 B", "Object B"],
  target: ["目标图片", "Target photo"],
  small: ["小图", "Preview"],
  normal: ["正常参考", "Normal reference"],
  raw: ["原图", "Original"],
  imageFailed: [
    "图片加载失败，请刷新。",
    "Photo failed to load. Please refresh.",
  ],
  corner: ["再点区域的对角。", "Click the opposite corner."],
  smallRegion: [
    "区域太小，请重新点两角。",
    "Region too small. Click two corners again.",
  ],
  regionDone: [
    "蓝框：你标出的细节（可选）",
    "Blue box: your marked detail (optional)",
  ],
  cropHint: [
    "在图上点区域的两个对角，裁剪；再点“裁剪”可取消。",
    "Click two opposite corners to crop. Click “Crop” again to cancel.",
  ],
  markHint: [
    "在图上点两个对角，标出你关注的细节；再点“画框”可取消。",
    "Click two opposite corners to mark a detail. Click “Mark a region” again to cancel.",
  ],
  zoomIn: ["放大", "Zoom in"],
  zoomOut: ["缩小", "Zoom out"],
  crop: ["裁剪", "Crop"],
  mark: ["画框（可选）", "Mark a region (optional)"],
  clearMark: ["清除我的框", "Clear my mark"],
  restore: ["恢复整图", "Restore whole photo"],
  ocr: ["识别文字", "Read text"],
  targetRequest: ["目标", "Target"],
  overview: ["概览照片", "Overview"],
  view: ["视角 {0}", "View {0}"],
  organizerInfo: [
    "20 人，每人 3 道搜证 + 2 道审核。已完成：{0} / 100。",
    "20 people, 3 search tasks + 2 reviews each. Completed: {0} / 100.",
  ],
  invitation: ["打开 {0} 的邀请链接", "Open invitation for {0}"],
  saved: [
    "已保存。可以继续下一题。",
    "Saved. You can continue to the next task.",
  ],
  progress: ["第 {0} / {1} 题 · {2}", "Task {0} / {1} · {2}"],
  review: ["图片审核", "Photo review"],
  search: ["搜证", "Evidence search"],
  steps: ["已操作 {0} 步", "{0} actions taken"],
  same: ["是同一实物", "Same physical object"],
  different: ["不是同一实物", "Different physical objects"],
  supported: ["声明成立", "Claim supported"],
  refuted: ["声明不成立", "Claim refuted"],
  uncertain: ["无法确定", "Cannot determine"],
  stateContext: [
    "正常参考来自另一个实物，不是目标的之前照片。",
    "The normal reference is a different object, not a before photo of this target.",
  ],
  identityContext: [
    "比较“对象 A”与“对象 B”的实物。橙色虚线框提示目标；不要只依据背景。",
    "Compare Object A with Object B. Orange dashed boxes indicate the target where provided; do not rely on backgrounds alone.",
  ],
  previewPhase: [
    "仅看这些小图，能判断吗？",
    "Can you decide from these previews?",
  ],
  limitedPhase: [
    "缺少原图时，这些照片够吗？",
    "Are these photos enough without the original?",
  ],
  sufficientPhase: [
    "这一组图片够判断吗？",
    "Is this set of photos enough to decide?",
  ],
  fullPhase: [
    "现在有全部可获取照片，能判断吗？",
    "You now have all available photos. Can you decide?",
  ],
  annotationPhase: [
    "确认最终判断和依据",
    "Confirm your final decision and evidence",
  ],
  initial: [
    "本题只有这些图片。",
    "These are the photos available for this task.",
  ],
  interactive: [
    "可以查看图片，或请求其他照片。",
    "You can view these photos or request others.",
  ],
  full: ["所有可获取照片都在这里。", "All available photos are shown here."],
  saveStage: ["保存，看下一组图片", "Save and see the next photos"],
  submit: ["提交本题", "Submit"],
  tooLarge: [
    "该图已经很大，可以先裁剪局部再放大。",
    "This photo is already large. Crop a region before zooming in further.",
  ],
  actionError: ["操作未成功：{0}", "Action unsuccessful: {0}"],
  unavailable: [
    "无法提供这张照片；这不能说明声明是真是假。",
    "This photo cannot be provided. That does not establish whether the claim is true.",
  ],
  textResult: ["识别文字：{0}", "Detected text: {0}"],
  actionDone: ["操作完成。", "Action completed."],
  agreeError: ["请先确认自愿参与。", "Please confirm your consent first."],
  verdictError: ["请选择判断。", "Please choose a decision."],
  reasonError: ["请简短写出理由。", "Please write a short reason."],
  copied: ["20 个个人邀请链接已复制。", "20 invitation links copied."],
  requestFailed: ["请求失败。", "Request failed."],
};
const STATIC_TEXT = {
  "header > span:first-of-type": "header",
  "#welcome .eyebrow": "welcomeEyebrow",
  "#welcome h1": "welcomeTitle",
  "#welcome > p:nth-of-type(2)": "welcomePurpose",
  "#welcome > p:nth-of-type(3)": "welcomeTasks",
  "#welcome .info": "inviteInfo",
  "#welcome summary": "demo",
  "#welcome details p": "demoNote",
  "#intro h1": "introTitle",
  "#intro > p": "introCount",
  "#intro li:nth-child(1)": "introFact",
  "#intro li:nth-child(2)": "introClick",
  "#intro li:nth-child(3)": "introZoom",
  "#intro li:nth-child(4)": "introReview",
  "#consent-text": "consent",
  "#begin": "begin",
  ".original-claim summary": "original",
  "#answer-title": "answer",
  "#request-panel summary": "requestPanel",
  ".optional summary": "optional",
  "#confidence-label": "confidence",
  "#reason-label": "reason",
  "#clear-label": "clear",
  "#checked-label": "checked",
  "#next": "next",
  "#done h1": "done",
  "#done p": "doneNote",
  "#preview-notice": "previewNotice",
  "#preview-link": "previewLink",
  "#admin h1": "admin",
  "#admin > p:first-of-type": "adminNote",
  "#refresh": "refresh",
  "#export": "export",
  "#votes": "votes",
  "#copy-links": "copyLinks",
  "#admin th:nth-child(1)": "id",
  "#admin th:nth-child(2)": "complete",
  "#admin th:nth-child(3)": "personal",
  footer: "footer",
  "#confidence option[value='1']": "low1",
  "#confidence option[value='2']": "low2",
  "#confidence option[value='3']": "mid",
  "#confidence option[value='4']": "high4",
  "#confidence option[value='5']": "high5",
  "#uncertain-label": "uncertain",
};
function t(key, ...args) {
  let value = UI_TEXT[key]?.[language === "en" ? 1 : 0] || key;
  args.forEach((arg, i) => {
    value = value.replaceAll("{" + i + "}", String(arg));
  });
  return value;
}
function applyLanguage() {
  document.documentElement.lang = language === "en" ? "en" : "zh-CN";
  document.title = t("title");
  for (const [selector, key] of Object.entries(STATIC_TEXT)) {
    const node = document.querySelector(selector);
    if (node) node.textContent = t(key);
  }
  document.getElementById("language").textContent =
    language === "en" ? "中文" : "English";
  document.getElementById("reason").placeholder = t("reasonPlaceholder");
}
function localizedError(text) {
  if (language === "zh") return text;
  const errors = {
    邀请链接无效或无权访问: "Invalid invitation or access denied.",
    "请求无效，请刷新当前题":
      "Invalid request. Please refresh the current task.",
    请选择判断: "Please choose a decision.",
    请简短写出判断理由: "Please write a short reason.",
    请选择把握程度: "Please choose a confidence level.",
    确定判断需要选择目标图片作为证据:
      "Select a target photo as evidence for a definite decision.",
    "身份判断请选择 A 和 B 各一张图片": "Select one photo of A and one of B.",
    证据框超出可见图片范围: "The evidence region is outside the visible photo.",
    "页面已更新，请刷新当前题": "This page is out of date. Please refresh.",
    审核阶段已更新: "This review stage has already been saved.",
    这组任务只查看已给出的图片: "This task uses only the provided photos.",
    此操作不可用: "This action is unavailable.",
    请先阅读说明: "Please read the instructions first.",
  };
  return errors[text] || text;
}
