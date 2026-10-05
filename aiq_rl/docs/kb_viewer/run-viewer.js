const T = i => D.texts[i];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
const $ = id => document.getElementById(id);
const snapBy = Object.fromEntries(D.snaps.map(s => [s.label, s]));
const lab = e => "kb_" + String(e).padStart(4, "0");
const scoreTag = s => `<span class="tag ${s >= 1 ? "ok" : s > 0 ? "mid" : "bad"}">得分 ${(+s).toFixed(2)}</span>`;
const linkC = s => esc(s).replace(/\bK\d{4}\b/g, m => `<a href="#" title="点击查看该条 claim 的演化（浮动窗口）" onclick="openClaim('${m}',event);return false">${m}</a>`);
let state = {tab: "overview", epoch: 1, q: null, sub: "records", snap: D.snaps[D.snaps.length - 1].label};

function nav() {
  const tabs = [["overview", "概览"], ["epoch", "每个 epoch：解题 / 发现 / 总结"], ["kb", "KB 快照"], ["claim", "单条 claim 演化"], ["commits", "commit 日志"], ["prompts", "提示词与配置"]];
  $("nav").innerHTML = tabs.map(([k, v]) => `<button class="${state.tab === k ? "on" : ""}" onclick="go('${k}')">${v}</button>`).join("");
  $("title").textContent = "KB 科学循环 · " + D.run;
  if (D.home) { $("home").href = D.home; } else { $("home").style.display = "none"; }
  $("meta").textContent = `模型 ${D.config.provider}/${D.config.model} · 每轮 ${D.config.batch_size} 题 · 共 ${D.epochs.length} 个 epoch`;
}
function go(t) { state.tab = t; render(); }

function render() {
  nav();
  ({overview, epoch: epochView, kb: kbView, claim: claimView, commits: commitView, prompts: promptView})[state.tab]();
}

/* ---------------- overview ---------------- */
function overview() {
  $("side").innerHTML = "";
  const ev = Object.entries(D.evals).sort();
  const rows = D.epochs.map(e => {
    const m = e.metrics, ops = Object.entries(m.ops || {}).filter(([k]) => k !== "credit").map(([k, v]) => `${k}:${v}`).join(" ");
    return `<tr><td><a href="#" onclick="state.epoch=${e.epoch};go('epoch');return false">e${e.epoch}</a></td><td>${(m.mean_score ?? 0).toFixed(3)}</td><td>${m.hypotheses ?? ""}</td><td>${m.kb_before ?? ""}→${m.kb_after ?? ""}</td><td>${ops}</td><td>${e.summarizer.length}</td></tr>`;
  }).join("");
  $("view").innerHTML = `
  <div class="card"><h3>流程</h3><div class="flow">
   <div style="border-left:4px solid var(--blue)"><b>① 解题</b><br>只看题目 + 只读 KB（快照 t−1 的渲染：cred top-40 + 试用 10）。无工具，effort=low。输出 answer + cited。</div>
   <div style="border-left:4px solid var(--amber)"><b>② 发现</b><br>同一会话续写：揭示答案与得分；工具 search_history / read_record / python 沙箱（只看前序 epoch）。对→low 可 skip，错→medium。输出 comment + hypothesis。</div>
   <div style="border-left:4px solid var(--violet)"><b>③ 总结者</b><br>epoch 末，xhigh。看本 epoch 全部记录 + 历史 + KB 索引；add / merge / delete / adjust，每个操作一条 commit；finish 写 science digest。</div>
   <div style="border-left:4px solid var(--green)"><b>④ KB 快照 t</b><br>credit：被引用的 claim 按得分记 s += score, f += 1−score；加上总结者的结构操作。</div></div></div>
  <div class="card"><h3>每个 epoch</h3><table><tr><th>epoch</th><th>本轮训练题均分</th><th>猜想数</th><th>KB 条数</th><th>结构操作</th><th>总结者轮数</th></tr>${rows}</table></div>
  <div class="card"><h3>评测</h3><table><tr><th>KB 版本</th><th>题数</th><th>均分</th><th>分题源</th></tr>${ev.map(([k, v]) => `<tr><td>${k}</td><td>${v.n}</td><td>${v.mean.toFixed(3)}</td><td>${esc(JSON.stringify(v.by_source))}</td></tr>`).join("")}</table></div>`;
}

/* ---------------- epoch ---------------- */
function epochView() {
  const E = D.epochs.find(e => e.epoch === state.epoch) || D.epochs[0];
  let side = D.epochs.map(e => `<div class="item ${e.epoch === E.epoch ? "on" : ""}" onclick="state.epoch=${e.epoch};state.q=null;render()">第 ${e.epoch} 轮 <small>均分 ${(e.metrics.mean_score ?? 0).toFixed(2)}</small></div>`).join("");
  if (state.sub === "records") {
    side += `<div class="note" style="padding:6px 10px">本轮题目</div>` + E.records.map((r, i) =>
      `<div class="item ${state.q === i ? "on" : ""}" onclick="state.q=${i};render()">${r.question_id} ${scoreTag(r.score)}<br><small>${r.source} · ${r.family} · ${r.skipped ? "跳过" : r.hypothesis ? "有猜想" : "仅评论"}</small></div>`).join("");
  }
  $("side").innerHTML = side;
  const sub = [["records", "解题 + 发现（逐题）"], ["summ", "总结者（逐轮）"], ["diff", "KB 变化"], ["sci", "科学总结"]];
  let h = `<div class="sub">${sub.map(([k, v]) => `<button class="${state.sub === k ? "on" : ""}" onclick="state.sub='${k}';render()">${v}</button>`).join("")}</div>`;
  if (state.sub === "records") h += state.q == null ? recordsTable(E) : recordView(E, E.records[state.q]);
  if (state.sub === "summ") h += summView(E);
  if (state.sub === "diff") h += diffView(lab(E.epoch - 1), lab(E.epoch), E.epoch);
  if (state.sub === "sci") h += `<div class="card summ"><h3>科学总结 · 第 ${E.epoch} 轮</h3><pre class="tall">${linkC(E.science)}</pre></div>`;
  $("view").innerHTML = h;
}
function recordsTable(E) {
  return `<div class="card"><h3>第 ${E.epoch} 轮：${E.records.length} 题（点左侧或下表查看完整内容）</h3><table><tr><th>题号</th><th>题源</th><th>得分</th><th>预测 / 答案</th><th>引用</th><th>发现阶段</th><th>评论</th></tr>${E.records.map((r, i) =>
    `<tr><td><a href="#" onclick="state.q=${i};render();return false">${r.question_id}</a></td><td>${r.source}</td><td>${scoreTag(r.score)}</td><td>${esc(r.prediction)}<br><small>${esc(r.answer)}</small></td><td>${linkC((r.cited || []).join(" "))}</td><td>${r.discover_effort}${r.skipped ? " 跳过" : ""} · ${r.discover_trace.length} 轮${r.hypothesis ? " · <b>有猜想</b>" : ""}</td><td>${esc(r.comment || "")}</td></tr>`).join("")}</table></div>`;
}
function turns(tr) {
  return tr.map(t => `<div class="turn"><b>第 ${t.turn + 1} 步</b> <span class="note">${t.secs ?? ""} 秒 · 输入 ${t.usage.prompt_tokens ?? "?"} token · 输出 ${t.usage.completion_tokens ?? "?"} token</span>
   ${t.reasoning ? `<details><summary>思维链</summary><pre>${esc(t.reasoning)}</pre></details>` : ""}
   ${t.content ? `<pre>${linkC(t.content)}</pre>` : ""}
   ${t.calls.map((c, j) => `<div class="call"><span class="nm">${esc(c.name)}</span><pre>${linkC(prettyArgs(c.arguments))}</pre><details ${j < 0 ? "open" : ""}><summary>工具返回</summary><pre>${linkC(t.results[j] ?? "")}</pre></details></div>`).join("")}</div>`).join("");
}
function prettyArgs(a) {
  try { const o = JSON.parse(a); if (o.code) return o.code + (Object.keys(o).length > 1 ? "\n# " + JSON.stringify({...o, code: undefined}) : ""); return JSON.stringify(o, null, 1); } catch (e) { return a; }
}
function recordView(E, r) {
  const snap = snapBy[lab(E.epoch - 1)];
  const cites = (r.cited || []).map(c => { const x = snap && snap.claims.find(z => z[0] === c); return x ? `<tr><td>${linkC(c)}</td><td>${x[3].toFixed(2)}</td><td>${x[1]}/${x[2]}</td><td>${esc(T(x[4]))}</td></tr>` : `<tr><td>${c}</td><td colspan=3>(不在快照中)</td></tr>`; }).join("");
  return `
  <div class="card"><h3>${r.question_id} · ${r.source} · ${r.task} · ${r.family}</h3>${scoreTag(r.score)} 预测 <b>${esc(r.prediction)}</b> · 答案 <b>${esc(r.answer)}</b>${r.inversions != null ? ` · 逆序对 ${r.inversions}` : ""}
   <details><summary>题目全文（解题者看到的 user/system 原文）</summary><pre>${esc(r.question)}</pre></details></div>
  <div class="card solver"><h3>① 解题 agent（effort=low，不给工具）</h3>
   <details><summary>注入的 KB（快照 ${snap ? snap.label : "?"}，${snap ? snap.claims.filter(c => c[6]).length : "?"} 条，追加在 system prompt 后；内容为英文原文）</summary><pre>${linkC(snap ? snap.render : "")}</pre></details>
   ${r.solver_reasoning ? `<p><b>思维链（luna 为摘要标题，Qwen 为完整思维链）</b></p><pre>${esc(r.solver_reasoning)}</pre>` : `<p class="note">该调用没有返回思维链。</p>`}
   <p><b>完整输出</b> <span class="note">${r.solve_secs} 秒 · ${esc(JSON.stringify(r.solve_usage))}</span></p><pre>${linkC(r.solution)}</pre>
   <p><b>引用的条目（解题时的状态）</b></p><table><tr><th>编号</th><th>可信度</th><th>胜/负</th><th>内容</th></tr>${cites || "<tr><td colspan=4>未引用</td></tr>"}</table></div>
  <div class="card disc"><h3>② 发现 agent（effort=${r.discover_effort}${r.skipped ? "，选择跳过" : ""}，${r.discover_trace.length} 轮）</h3>
   ${turns(r.discover_trace)}
   <p><b>评论（comment）</b></p><pre>${linkC(r.comment || "-")}</pre>
   <p><b>猜想（hypothesis）</b></p><pre>${linkC(r.hypothesis || "-")}</pre></div>`;
}
function summView(E) {
  const ops = D.commits.filter(c => c.epoch === E.epoch && !["credit", "seed"].includes(c.op));
  return `<div class="card summ"><h3>③ 总结者 · 第 ${E.epoch} 轮（${E.summarizer.length} 步，effort=xhigh）</h3>
   <p><b>本轮 KB 结构操作</b>（${ops.length}）</p>${commitTable(ops)}
   ${turns(E.summarizer)}</div>`;
}

/* ---------------- KB diff / snapshot ---------------- */
function diffView(a, b, epoch) {
  const A = snapBy[a], B = snapBy[b];
  if (!A || !B) return `<div class="card">缺少快照 ${a} / ${b}</div>`;
  const ma = Object.fromEntries(A.claims.map(c => [c[0], c])), mb = Object.fromEntries(B.claims.map(c => [c[0], c]));
  const added = B.claims.filter(c => !ma[c[0]]), removed = A.claims.filter(c => !mb[c[0]]);
  const changed = B.claims.filter(c => ma[c[0]] && ma[c[0]][4] !== c[4]);
  const credit = B.claims.filter(c => ma[c[0]] && ma[c[0]][4] === c[4] && (ma[c[0]][1] !== c[1] || ma[c[0]][2] !== c[2]));
  const inR = B.claims.filter(c => c[6] && !(ma[c[0]] && ma[c[0]][6])), outR = A.claims.filter(c => c[6] && !(mb[c[0]] && mb[c[0]][6]));
  const row = (c, cls, extra = "") => `<tr class="${cls}"><td>${linkC(c[0])}</td><td>${c[1]}/${c[2]}</td><td>${c[3].toFixed(2)}</td><td>${extra}${esc(T(c[4]))}</td></tr>`;
  const tb = (rows) => `<table><tr><th>编号</th><th>胜/负</th><th>可信度</th><th>内容</th></tr>${rows || "<tr><td colspan=4>无</td></tr>"}</table>`;
  return `<div class="card kbc"><h3>KB 变化 ${a} → ${b}：${A.claims.length} → ${B.claims.length} 条</h3>
   <p>新增 ${added.length} · 删除或被合并 ${removed.length} · 文本改写 ${changed.length} · 仅胜负计数变化 ${credit.length} · 新进入解题者视野 ${inR.length} · 移出视野 ${outR.length}</p></div>
   <div class="card kbc"><h3>新增</h3>${tb(added.map(c => row(c, "add")).join(""))}</div>
   <div class="card kbc"><h3>删除 / 被合并</h3>${tb(removed.map(c => row(c, "del")).join(""))}</div>
   <div class="card kbc"><h3>文本改写（上：旧，下：新）</h3>${tb(changed.map(c => row(c, "chg", `<div class="note">${esc(T(ma[c[0]][4]))}</div><hr>`)).join(""))}</div>
   <div class="card kbc"><h3>引用记账（胜/负计数变化）</h3><table><tr><th>编号</th><th>之前</th><th>之后</th><th>可信度</th><th>内容</th></tr>${credit.map(c => `<tr><td>${linkC(c[0])}</td><td>${ma[c[0]][1]}/${ma[c[0]][2]}</td><td>${c[1]}/${c[2]}</td><td>${ma[c[0]][3].toFixed(2)}→${c[3].toFixed(2)}</td><td>${esc(T(c[4]).slice(0, 200))}</td></tr>`).join("")}</table></div>
   <div class="card kbc"><h3>解题者视野变化（下一轮注入的 50 条）</h3><p><b>进入</b></p>${tb(inR.map(c => row(c, "add")).join(""))}<p><b>掉出</b></p>${tb(outR.map(c => row(c, "del")).join(""))}</div>`;
}
function kbView() {
  $("side").innerHTML = D.snaps.map(s => `<div class="item ${s.label === state.snap ? "on" : ""}" onclick="state.snap='${s.label}';render()">${s.label} <small>${s.claims.length} 条</small></div>`).join("");
  const S = snapBy[state.snap];
  $("view").innerHTML = `<div class="sub"><button class="${state.sub !== "render" ? "on" : ""}" onclick="state.sub='table';render()">全部条目</button><button class="${state.sub === "render" ? "on" : ""}" onclick="state.sub='render';render()">解题者看到的 KB 原文</button></div>` +
    (state.sub === "render" ? `<div class="card solver"><h3>${S.label} 注入原文（下一轮解题者看到的）</h3><pre class="tall">${linkC(S.render)}</pre></div>` :
    `<div class="card kbc"><h3>${S.label}：${S.claims.length} 条，绿色左边线 = 在解题者视野中（${S.claims.filter(c => c[6]).length} 条）</h3>
     <input type="search" placeholder="搜索内容，例如 ReLU / SiLU / RMSprop" oninput="filt(this.value)">
     <table id="kbt"><tr><th>编号</th><th>可信度</th><th>胜/负</th><th>来源</th><th>内容</th></tr>${S.claims.slice().sort((x, y) => y[3] - x[3]).map(c =>
      `<tr class="${c[6] ? "shown" : ""}" data-t="${esc(T(c[4]).toLowerCase())}"><td>${linkC(c[0])}</td><td><span class="bar"><i style="width:${c[3] * 100}%"></i></span> ${c[3].toFixed(2)}</td><td>${c[1]}/${c[2]}</td><td>${c[5] ? "第 " + c[5] + " 轮" : "种子"}</td><td>${esc(T(c[4]))}</td></tr>`).join("")}</table></div>`);
}
function filt(v) { v = v.toLowerCase(); document.querySelectorAll("#kbt tr[data-t]").forEach(r => r.style.display = r.dataset.t.includes(v) ? "" : "none"); }

/* ---------------- claim history ---------------- */
let claimId = "K0134";
function goClaim(id) { claimId = id; state.tab = "claim"; render(); }
function openClaim(id, ev) {
  let rows = "", last = null;
  for (const s of D.snaps) {
    const c = s.claims.find(z => z[0] === id);
    if (!c) { rows += `<tr><td>${s.label}</td><td colspan=3 class="note">不存在</td></tr>`; continue; }
    rows += `<tr class="${c[6] ? "shown" : ""}"><td>${s.label}</td><td>${c[1]}/${c[2]}</td><td><span class="bar"><i style="width:${c[3] * 100}%"></i></span> ${c[3].toFixed(2)}${c[6] ? " · 视野中" : ""}</td><td>${c[4] !== last ? esc(T(c[4])) : '<span class="note">（文本同上）</span>'}</td></tr>`;
    last = c[4];
  }
  const cs = D.commits.filter(c => c.op !== "credit" && ((c.claim_ids || []).includes(id) || c.new_id === id));
  $("popc").innerHTML = `<h3>${esc(id)} 浮动详情 <a href="#" style="font-weight:400;font-size:12px" onclick="closePop();goClaim('${id}');return false">在「单条 claim 演化」页打开 →</a><button id="popx" onclick="closePop()">关闭 ✕</button></h3>
    <table><tr><th>快照</th><th>胜/负</th><th>可信度</th><th>内容（文本变化时显示）</th></tr>${rows || `<tr><td colspan=4 class="note">从未在任何快照中出现</td></tr>`}</table>
    <h3 style="margin-top:10px">结构操作（${cs.length}）</h3>${commitTable(cs)}`;
  const p = $("pop"), card = $("popc");
  p.style.display = "block";
  const w = card.offsetWidth, hh = Math.min(card.offsetHeight, innerHeight * .84);
  card.style.left = Math.min(Math.max((ev?.clientX ?? innerWidth / 2) - w / 2, 12), Math.max(innerWidth - w - 12, 12)) + "px";
  card.style.top = Math.min(Math.max((ev?.clientY ?? 80) - 60, 12), Math.max(innerHeight - hh - 12, 12)) + "px";
  card.scrollTop = 0;
}
function closePop() { $("pop").style.display = "none"; }
document.addEventListener("keydown", e => { if (e.key === "Escape") closePop(); });
function claimView() {
  const ids = [...new Set(D.snaps.flatMap(s => s.claims.map(c => c[0])))].sort();
  $("side").innerHTML = `<div style="padding:6px"><input type="search" placeholder="K0134" onchange="goClaim(this.value.trim().toUpperCase())"></div>` +
    ids.map(i => `<div class="item ${i === claimId ? "on" : ""}" onclick="goClaim('${i}')">${i}</div>`).join("");
  let last = null, rows = "";
  for (const s of D.snaps) {
    const c = s.claims.find(z => z[0] === claimId);
    if (!c) { rows += `<tr><td>${s.label}</td><td colspan=4 class="note">不存在</td></tr>`; continue; }
    rows += `<tr class="${c[6] ? "shown" : ""}"><td>${s.label}</td><td>${c[1]}/${c[2]}</td><td><span class="bar"><i style="width:${c[3] * 100}%"></i></span> ${c[3].toFixed(2)}</td><td>${c[6] ? "视野中" : ""}</td><td>${c[4] !== last ? esc(T(c[4])) : '<span class="note">（文本同上）</span>'}</td></tr>`;
    last = c[4];
  }
  const cs = D.commits.filter(c => (c.claim_ids || []).includes(claimId) || c.new_id === claimId);
  const cr = cs.filter(c => c.op === "credit"), st = cs.filter(c => c.op !== "credit");
  $("view").innerHTML = `<div class="card kbc"><h3>${claimId} 在各快照中的状态</h3><table><tr><th>快照</th><th>胜/负</th><th>可信度</th><th></th><th>内容</th></tr>${rows}</table></div>
   <div class="card summ"><h3>结构操作 commit（${st.length}）</h3>${commitTable(st)}</div>
   <div class="card"><h3>引用记账 commit（${cr.length}）</h3>${commitTable(cr)}</div>`;
}

/* ---------------- commits ---------------- */
function commitTable(cs) {
  if (!cs.length) return `<p class="note">无</p>`;
  return `<table><tr><th>commit</th><th>轮次</th><th>操作</th><th>条目</th><th>来源题</th><th>内容 / 理由</th></tr>${cs.map(c => {
    const p = c.payload || {};
    const body = (p.text ? `<b>新内容：</b> ${linkC(p.text)}<br>` : "") + (p.old_text ? `<span class="note"><b>旧内容：</b> ${esc(p.old_text)}</span><br>` : "") +
      (c.op === "credit" ? esc(JSON.stringify(p)) : "") + (c.reason ? `<b>理由：</b> ${linkC(c.reason)}` : "");
    return `<tr class="${c.op === "add" ? "add" : c.op === "delete" ? "del" : ["merge", "revise", "rollback"].includes(c.op) ? "chg" : ""}"><td>${c.commit}</td><td>${c.epoch}</td><td>${c.op}</td><td>${linkC((c.claim_ids || []).join(" "))}${c.new_id ? " → " + linkC(c.new_id) : ""}</td><td>${esc((c.source_question_ids || []).join(" "))}</td><td>${body}</td></tr>`;
  }).join("")}</table>`;
}
function commitView() {
  const ops = [...new Set(D.commits.map(c => c.op))];
  $("side").innerHTML = ["(结构操作)", ...ops].map(o => `<div class="item ${state.op === o ? "on" : ""}" onclick="state.op='${o}';render()">${o} <small>${o.startsWith("(") ? D.commits.filter(c => !["credit", "seed"].includes(c.op)).length : D.commits.filter(c => c.op === o).length}</small></div>`).join("");
  const o = state.op || "(结构操作)";
  const cs = o.startsWith("(") ? D.commits.filter(c => !["credit", "seed"].includes(c.op)) : D.commits.filter(c => c.op === o);
  $("view").innerHTML = `<div class="card"><h3>commit 日志 · ${o}（${cs.length} 条）</h3>${commitTable(cs)}</div>`;
}

/* ---------------- prompts ---------------- */
function promptView() {
  $("side").innerHTML = "";
  $("view").innerHTML = Object.entries(D.prompts).map(([k, v]) => `<div class="card"><h3>${esc(k)}</h3><pre class="tall">${esc(v)}</pre></div>`).join("") +
    `<div class="card"><h3>运行配置</h3><pre>${esc(JSON.stringify(D.config, null, 1))}</pre></div>`;
}
render();
