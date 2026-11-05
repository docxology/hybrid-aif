(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var B = null, S = { setting: null, t: 0, playing: false, timer: null, tier: "rover", showAll: false };
  var $ = function (id) { return document.getElementById(id); };
  var TOP_N = 10, CACHE = {}, WIN = null;

  function el(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function fmt(x, d) { return (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d == null ? 2 : d); }
  function pretty(s) { return String(s).replace(/_/g, " "); }
  function short(n) { var i = n.indexOf("."); return i < 0 ? n : n.slice(i + 1); }
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; }); }
  function names(k) { return (B.meta.names && B.meta.names[k]) || []; }
  function nm(k, i) { return names(k)[i] != null ? names(k)[i] : String(i); }
  function argmax(a) { var b = 0; for (var i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

  // ---- loading: after first paint, bundle.js by script tag (works over http and file://) ----
  function load() {
    return new Promise(function (res, rej) {
      if (window.HYBRID_AIF_BUNDLE) return res(window.HYBRID_AIF_BUNDLE);
      var s = document.createElement("script");
      s.src = "bundle.js";
      s.onload = function () { window.HYBRID_AIF_BUNDLE ? res(window.HYBRID_AIF_BUNDLE) : rej(new Error("bundle.js did not define the replay data")); };
      s.onerror = function () { rej(new Error("could not load bundle.js")); };
      document.head.appendChild(s);
    });
  }

  function run() { return B.runs[S.setting]; }
  function tick() { return run().ticks[S.t]; }
  function cache() {
    var c = CACHE[S.setting];
    if (c) return c;
    var T = run().ticks, last = [], segs = [], d = -1;
    T.forEach(function (t, i) {
      if (t.slow_policies && t.slow_policies.length) { d = i; segs.push({ t0: i, opt: t.option }); }
      last.push(d);
    });
    segs.forEach(function (s, i) { s.t1 = i + 1 < segs.length ? segs[i + 1].t0 : T.length; });
    return (CACHE[S.setting] = { last: last, segs: segs, range: {} });
  }
  function decTick() { var i = cache().last[S.t]; return i < 0 ? null : run().ticks[i]; }

  // ---- colours ----
  var TERM_VAR = { risk: "--c1", uncertainty: "--c2", ambiguity: "--c5", infogain: "--c3", utility: "--c4" };
  function termColor(t) { return css(TERM_VAR[t] || "--muted"); }
  function optColor(i) { return css("--o" + i); }
  function optAlpha(i) { return i < 3 ? 0.45 : i < 6 ? 0.75 : 0.6; }
  function qualColor(q) { return css(["--qg", "--qb", "--qs"][q]); }
  function optZone(i) {
    var n = nm("option", i); return /_(\d)$/.test(n) ? "rock_" + n.slice(-1) : "dock";
  }
  function optLabel(i) { return pretty(nm("option", i)); }
  function dur(i) { var c = durations(); return c ? c[i] : null; }
  function durations() {
    var d = null;
    B.graph.tiers.forEach(function (t) { t.controls.forEach(function (c) { if (c.durations) d = c.durations; }); });
    return d;
  }

  // ---- graph ----
  var ALIAS = { "mission.option": "mission.u_option", "rover.a_sensor": "rover.sensor" };
  var FAM = { categorical: "categorical", gaussian_mixture: "Gauss. mixture", linear_gaussian: "lin. Gaussian", neural: "neural (MLP)" };
  function encoderSetting() { var s = settingRec(); return !!(s && s.camera === "encoder"); }
  function settingRec() { for (var i = 0; i < B.settings.length; i++) if (B.settings[i].id === S.setting) return B.settings[i]; return null; }
  function showInfogain() {
    var r = settingRec(), host = $("infogain"); if (!host) return;
    host.textContent = !r || !r.infogain ? "" : "Rover tier information gain scored on: " + (r.infogain.length ? r.infogain.map(function (m) { return m === "img" ? "camera" : m; }).join(", ") + ". The mission tier scores the rock checks in every epistemic setting." : "none (epistemic terms off in both tiers).");
  }
  function drawGraph() {
    showInfogain();
    var svg = $("graph"); svg.innerHTML = "";
    var W = 480, LH = 80, MAXPER = 4;
    var kinds = {
      transition: { dash: "", color: "--fg", w: 1.3 },
      likelihood: { dash: "5 3", color: "--c1", w: 1.3 },
      target: { dash: "2 3", color: "--c2", w: 2.2 },
      binding: { dash: "7 3 2 3", color: "--c3", w: 1.6 },
      recognition: { dash: "1 3", color: "--c4", w: 2 }
    };
    var defs = el("defs", {}, svg);
    for (var k in kinds) {
      var m = el("marker", { id: "ar-" + k, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", fill: css(kinds[k].color) }, m);
    }
    var pos = {}, info = {}, y = 10, boxes = [];
    B.graph.tiers.forEach(function (t, ti) {
      var rows = [];
      if (t.controls.length) rows.push(["control", t.controls]);
      if (t.factors.length) rows.push(["factor", t.factors]);
      if (t.modalities.length) rows.push(["modality", t.modalities]);
      var y0 = y; y += 26;
      rows.forEach(function (row) {
        var n = row[1].length, lines = Math.ceil(n / MAXPER), per = Math.ceil(n / lines);
        for (var l = 0; l < lines; l++) {
          var part = row[1].slice(l * per, (l + 1) * per);
          part.forEach(function (nd, i) {
            pos[nd.name] = { x: 8 + (W - 16) * (i + 0.5) / part.length, y: y + 34, sp: (W - 16) / part.length };
            info[nd.name] = { kind: row[0], nd: nd };
          });
          y += LH;
        }
      });
      boxes.push([y0, y - y0 - 4, t.name + " tier" + (ti === 0 ? " (slow)" : " (fast)")]);
      y += 12;
    });
    var H = y + 22; svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    boxes.forEach(function (b) {
      el("rect", { x: 8, y: b[0], width: W - 16, height: b[1], rx: 6, fill: "none", stroke: css("--line"), "stroke-width": 1.5 }, svg);
      el("text", { x: 18, y: b[0] + 16, "font-size": 12, "font-weight": 600, class: "muted" }, svg, b[2]);
    });
    var seen = {}, edges = B.graph.edges.slice();
    if (encoderSetting() && B.graph.recognition) B.graph.recognition.forEach(function (r) {
      r.parents.forEach(function (p) { edges.push({ from: p.replace(/^obs:/, ""), to: r.variable, kind: "recognition" }); });
    });
    edges.forEach(function (e) {
      var from = ALIAS[e.from] || e.from, to = ALIAS[e.to] || e.to, a = pos[from], b = pos[to];
      if (!a || !b) return;
      var key = from + ">" + to, nth = seen[key] = (seen[key] || 0) + 1;
      var st = kinds[e.kind] || kinds.transition;
      var attrs = { fill: "none", stroke: css(st.color), "stroke-width": st.w, "stroke-dasharray": st.dash, "marker-end": "url(#ar-" + e.kind + ")" };
      if (from === to) return;
      var dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
      var r1 = 28, r2 = 32;
      var x1 = a.x + ux * r1, y1 = a.y + uy * r1, x2 = b.x - ux * r2, y2 = b.y - uy * r2;
      var bend = nth === 1 ? 10 : -22;
      var cx = (x1 + x2) / 2 - uy * bend, cy = (y1 + y2) / 2 + ux * bend;
      el("path", Object.assign({ d: "M" + x1 + "," + y1 + " Q" + cx + "," + cy + " " + x2 + "," + y2 }, attrs), svg);
    });
    Object.keys(pos).forEach(function (name) {
      var p = pos[name], i = info[name], g = el("g", { "data-node": name }, svg);
      var fill = css("--bg"), stroke = css("--fg"), label = short(name), nd = i.nd, many = p.sp < 100;
      var fs = label.length > 9 ? 10 : 11.5;
      if (i.kind === "factor") el("circle", { cx: p.x, cy: p.y, r: 26, fill: fill, stroke: stroke, "stroke-width": 1.6 }, g);
      else if (i.kind === "modality") el("rect", { x: p.x - 30, y: p.y - 17, width: 60, height: 34, fill: fill, stroke: stroke, "stroke-width": 1.6 }, g);
      else el("rect", { x: p.x - 36, y: p.y - 15, width: 72, height: 30, rx: 15, fill: fill, stroke: css("--c3"), "stroke-width": 1.8 }, g);
      el("text", { x: p.x, y: p.y + 4, "text-anchor": "middle", "font-size": fs }, g, label);
      if (i.kind === "control") {
        var d = nd.durations;
        var dd = d ? d.map(function (v) { return Math.round(v); }) : null, same = dd && dd.every(function (v) { return v === dd[0]; });
        var s2 = dd ? (same ? dd[0] + " ticks each" : "ticks " + dd.join("/")) : (nd.kind === "level" ? nd.levels + " levels" : "continuous");
        el("text", { x: p.x, y: p.y + 28, "text-anchor": "middle", "font-size": 9.5, class: "muted", "font-weight": d ? 700 : 400 }, g, s2);
        el("text", { x: p.x, y: p.y + 40, "text-anchor": "middle", "font-size": 9.5, class: "now" }, g, "");
      } else {
        var sub = FAM[nd.family] || pretty(nd.family || "");
        if (i.kind === "factor") sub += nd.levels ? " · " + nd.levels : " · dim " + nd.dim;
        el("text", { x: p.x, y: p.y + (i.kind === "factor" ? 40 : 31), "text-anchor": "middle", "font-size": 9.5, class: "muted" }, g, sub);
      }
    });
    var lx = 16, ly = H - 8, present = {};
    edges.forEach(function (e) { present[e.kind] = true; });
    Object.keys(kinds).forEach(function (k) {
      if (!present[k]) return;
      var st = kinds[k];
      el("line", { x1: lx, y1: ly - 4, x2: lx + 22, y2: ly - 4, stroke: css(st.color), "stroke-width": st.w, "stroke-dasharray": st.dash }, svg);
      el("text", { x: lx + 27, y: ly, "font-size": 10, class: "muted" }, svg, k);
      lx += 27 + 7 * k.length + 14;
    });
    updateGraph();
  }
  function updateGraph() {
    var t = tick();
    document.querySelectorAll("#graph g[data-node]").forEach(function (g) {
      var n = g.getAttribute("data-node"), shape = g.firstElementChild, now = g.querySelector(".now");
      if (/u_option$/.test(n)) {
        shape.setAttribute("fill", optColor(t.option)); shape.setAttribute("fill-opacity", 0.35);
        if (now) now.textContent = "now: " + optLabel(t.option);
      } else if (/\.sensor$/.test(n)) {
        shape.setAttribute("fill", t.sensor ? css("--c3") : css("--bg")); shape.setAttribute("fill-opacity", t.sensor ? 0.35 : 1);
        if (now) now.textContent = "now: " + nm("sensor", t.sensor);
      }
    });
  }

  // ---- field canvas with a padded world window ----
  function computeWindow() {
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    function add(x, y, r) { x0 = Math.min(x0, x - r); x1 = Math.max(x1, x + r); y0 = Math.min(y0, y - r); y1 = Math.max(y1, y + r); }
    var m = B.meta;
    if (m.field) { add(m.field[0][0], m.field[1][0], 0); add(m.field[0][1], m.field[1][1], 0); }
    Object.keys(m.zones || {}).forEach(function (z) { add(m.zones[z][0], m.zones[z][1], 0.3); });
    if (m.beacon) add(m.beacon[0], m.beacon[1], 0.3);
    Object.keys(B.runs).forEach(function (k) {
      B.runs[k].ticks.forEach(function (t) {
        add(t.true_x[0], t.true_x[1], 0.15);
      });
    });
    var pad = 0.06 * Math.max(x1 - x0, y1 - y0);
    return { x0: x0 - pad, x1: x1 + pad, y0: y0 - pad, y1: y1 + pad };
  }
  function setupCanvas(c) {
    var dpr = window.devicePixelRatio || 1, w = c.clientWidth || 520;
    var sc = w / (WIN.x1 - WIN.x0), h = Math.round((WIN.y1 - WIN.y0) * sc);
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    var g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {
      g: g, w: w, h: h, sc: sc,
      X: function (x) { return (x - WIN.x0) * sc; },
      Y: function (y) { return h - (y - WIN.y0) * sc; }
    };
  }
  function halo(g, text, x, y) {
    g.lineWidth = 3; g.strokeStyle = css("--panel"); g.strokeText(text, x, y); g.fillStyle = css("--fg"); g.fillText(text, x, y);
  }
  function drawGrid(m) {
    var g = m.g;
    g.fillStyle = css("--panel"); g.fillRect(0, 0, m.w, m.h);
    g.strokeStyle = css("--line"); g.lineWidth = 1; g.beginPath();
    for (var x = Math.ceil(WIN.x0); x <= WIN.x1; x++) { g.moveTo(m.X(x), 0); g.lineTo(m.X(x), m.h); }
    for (var y = Math.ceil(WIN.y0); y <= WIN.y1; y++) { g.moveTo(0, m.Y(y)); g.lineTo(m.w, m.Y(y)); }
    g.stroke();
    var f = B.meta.field;
    if (f) { g.setLineDash([4, 4]); g.strokeStyle = css("--muted"); g.strokeRect(m.X(f[0][0]), m.Y(f[1][1]), (f[0][1] - f[0][0]) * m.sc, (f[1][1] - f[1][0]) * m.sc); g.setLineDash([]); }
  }
  function drawZones(m, t) {
    var g = m.g, zs = B.meta.zones || {}, sl = t.slow, target = optZone(t.option);
    g.font = "12px system-ui, sans-serif"; g.textAlign = "center"; g.lineJoin = "round";
    Object.keys(zs).forEach(function (z) {
      var x = m.X(zs[z][0]), y = m.Y(zs[z][1]), mm = /^rock_(\d)$/.exec(z), col, r = 10;
      if (z === target) { g.save(); g.setLineDash([4, 3]); g.strokeStyle = css("--accent"); g.lineWidth = 2; g.beginPath(); g.arc(x, y, 17, 0, 2 * Math.PI); g.stroke(); g.restore(); }
      g.lineWidth = 2;
      if (mm) {
        var q = argmax(sl.rocks_belief[+mm[1] - 1]); col = qualColor(q);
        g.fillStyle = col; g.globalAlpha = 0.35; g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill(); g.globalAlpha = 1;
        g.strokeStyle = col; g.stroke();
      } else if (z === "dock") {
        col = css("--c3"); g.fillStyle = col; g.globalAlpha = 0.3; g.fillRect(x - r, y - r, 2 * r, 2 * r); g.globalAlpha = 1; g.strokeStyle = col; g.strokeRect(x - r, y - r, 2 * r, 2 * r);
      } else {
        g.strokeStyle = css("--muted"); g.beginPath(); g.moveTo(x - 6, y); g.lineTo(x + 6, y); g.moveTo(x, y - 6); g.lineTo(x, y + 6); g.stroke();
      }
      halo(g, pretty(z), x, y + (z === "start" ? 20 : 27));
    });
    if (B.meta.beacon) {
      var bx = m.X(B.meta.beacon[0]), by = m.Y(B.meta.beacon[1]), c2 = css("--c2");
      g.beginPath(); g.moveTo(bx, by - 11); g.lineTo(bx + 9, by + 8); g.lineTo(bx - 9, by + 8); g.closePath();
      g.fillStyle = c2; g.globalAlpha = 0.25; g.fill(); g.globalAlpha = 1; g.strokeStyle = c2; g.lineWidth = 2; g.stroke();
      halo(g, "ridge (fix sharpest here)", bx, by + 22);
    }
  }
  function ellipse(m, comp, color, alpha) {
    var g = m.g, a = comp.cov[0][0], b = comp.cov[0][1], d = comp.cov[1][1];
    var tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    var l1 = tr / 2 + disc, l2 = Math.max(tr / 2 - disc, 1e-9), th = 0.5 * Math.atan2(2 * b, a - d);
    g.beginPath(); g.ellipse(m.X(comp.mu[0]), m.Y(comp.mu[1]), 2 * Math.sqrt(l1) * m.sc, 2 * Math.sqrt(l2) * m.sc, -th, 0, 2 * Math.PI);
    g.globalAlpha = alpha * 0.45; g.fillStyle = color; g.fill();
    g.globalAlpha = Math.min(1, alpha * 1.4 + 0.15); g.strokeStyle = color; g.lineWidth = 1.5; g.stroke();
    g.globalAlpha = 1;
  }
  function drawField() {
    var c = $("field"), m = setupCanvas(c), g = m.g, r = run().ticks, t = r[S.t];
    drawGrid(m); drawZones(m, t);
    t.belief.components.forEach(function (cm) { ellipse(m, cm, css("--c1"), cm.w); });
    g.strokeStyle = css("--fg"); g.lineWidth = 2; g.beginPath();
    for (var i = 0; i <= S.t; i++) { var p = r[i].true_x; i ? g.lineTo(m.X(p[0]), m.Y(p[1])) : g.moveTo(m.X(p[0]), m.Y(p[1])); }
    g.stroke();
    if (t.fix) {
      var fx = m.X(t.fix[0]), fy = m.Y(t.fix[1]);
      g.strokeStyle = css("--c2"); g.lineWidth = 2; g.beginPath();
      g.moveTo(fx - 5, fy - 5); g.lineTo(fx + 5, fy + 5); g.moveTo(fx - 5, fy + 5); g.lineTo(fx + 5, fy - 5); g.stroke();
    }
    g.beginPath(); g.arc(m.X(t.true_x[0]), m.Y(t.true_x[1]), 6, 0, 2 * Math.PI);
    g.fillStyle = css("--fg"); g.fill(); g.strokeStyle = css("--bg"); g.lineWidth = 2; g.stroke();
    drawBadges(t);
    var terr = t.belief.terrain, best = argmax(terr), v = t.belief.velocity;
    $("hud").innerHTML = "<span>t=<b>" + t.t + "</b></span><span>terrain: true <b>" + nm("terrain", t.true_terrain) +
      "</b>, belief " + nm("terrain", best) + " " + terr[best].toFixed(2) + "</span><span>components: <b>" +
      t.belief.components.length + "</b></span>" + (v ? "<span>belief velocity: <b>" + v[0].toFixed(2) + ", " + v[1].toFixed(2) + "</b></span>" : "") +
      (t.free_energy != null ? "<span>free energy: <b>" + t.free_energy.toFixed(2) + "</b></span>" : "");
    drawCamera(t);
  }
  function imageRows(x) {
    // an image as rows of 0..255: either rows already, or a base64 string of the pixel bytes
    if (typeof x !== "string") return x;
    var bytes = atob(x), side = (B.meta.image && B.meta.image.side) || Math.round(Math.sqrt(bytes.length)), rows = [];
    for (var i = 0; i < side; i++) { var row = []; for (var j = 0; j < side; j++) row.push(bytes.charCodeAt(i * side + j)); rows.push(row); }
    return rows;
  }
  function pixels(c, rows, overlay) {
    rows = imageRows(rows);
    var n = rows.length, dpr = window.devicePixelRatio || 1, sz = 160;
    c.width = sz * dpr; c.height = sz * dpr;
    var g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    var cell = sz / n;
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
      var v = rows[i][j]; g.fillStyle = "rgb(" + v + "," + v + "," + v + ")"; g.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
    }
    if (overlay) overlay(g, sz);
    return g;
  }
  function drawCamera(t) {
    var host = $("camgrid");
    if (!t.img) { $("camerafig").hidden = true; return; }
    $("camerafig").hidden = false;
    if (!host.children.length) {
      ["observed image", "decoder prediction at the belief", "terrain readout"].forEach(function (lbl, i) {
        var d = document.createElement("div"); d.className = "cam";
        d.innerHTML = (i < 2 ? "<canvas id='cam" + i + "' width='160' height='160' role='img' aria-label='" + lbl + "'></canvas>" : "<div class='readout' id='cam2' role='img' aria-label='" + lbl + "'></div>") +
          "<span class='cl'>" + lbl + "</span><span class='cv' id='camv" + i + "'></span>";
        host.appendChild(d);
      });
    }
    if (!$("terrlegend").children.length) {
      var tl = "<span>Terrain brightness:</span>";
      [0.15, 0.40, 0.65].forEach(function (v, i) { var g8 = Math.round(v * 255); tl += "<span><i style='background:rgb(" + g8 + "," + g8 + "," + g8 + ");border:1px solid " + css("--line") + "'></i>" + esc(nm("terrain", i)) + "</span>"; });
      $("terrlegend").innerHTML = tl;
    }
    var im = B.meta.image || {}, side = im.side || imageRows(t.img).length, cam = t.camera || {}, rec = Array.isArray(t.encoder) ? t.encoder : null, sr = settingRec(), ig = sr && sr.infogain;
    $("camnote").textContent = "A top-down patch of terrain around the rover, " + side + " pixels per side" + (im.half_width != null ? ", spanning \u00b1" + im.half_width + (im.half_width === 1 ? " field unit around it" : " field units around it") : "") +
      ". Brightness and stripe texture mark the terrain, and pixel noise is added. The patch carries no mark of the rover: it identifies the terrain and constrains position along the stripe gradient, through the decoder's linearisation; the position fix is the only modality that locates the rover on both axes.";
    pixels($("cam0"), t.img);
    pixels($("cam1"), t.recon || t.img);
    var rows = [["believed", t.belief.terrain]];
    if (rec) rows.push([encoderSetting() ? "recogniser (used)" : "recogniser (comparison)", rec]);
    var h = "";
    rows.forEach(function (r) {
      h += "<div class='rl'>" + esc(r[0]) + "</div>";
      r[1].forEach(function (p, i) {
        h += "<div class='rb" + (i === t.true_terrain ? " true" : "") + "'><span class='rn'>" + esc(nm("terrain", i)) + (i === t.true_terrain ? " (true)" : "") + "</span><span class='bar'><i style='width:" + (100 * p).toFixed(1) + "%'></i></span><span class='rp'>" + p.toFixed(2) + "</span></div>";
      });
    });
    $("cam2").innerHTML = h;
    $("camv1").textContent = cam.log_evidence != null ? "log evidence " + cam.log_evidence.toFixed(1) :
      cam.covered_term != null ? "covered term " + cam.covered_term.toFixed(1) + ", normaliser " + cam.normaliser.toFixed(1) : "";
    $("camcap").innerHTML = "The camera: the observed " + side + "&times;" + side + " patch (left), the decoder's prediction at the belief mean and most likely terrain (middle), and a terrain readout (right): the rover's belief over terrain, and the true terrain marked. " +
      (rec ? "The recogniser's probability vector for the image is shown beside the belief. " : "") +
      (encoderSetting() ? "In this setting the recogniser's probabilities replace the inversion of the image for the terrain factor; the covered term is the image's expected cost under the belief, and the normaliser is the measurement's evidence."
        : "In this setting the engine inverts the decoder at the belief; the number under the middle panel is the image's log evidence under that linearisation. The recogniser is run for comparison only.") +
      (ig ? (ig.indexOf("img") >= 0 ? " The planner scores the camera's information gain." : " The planner does not score the camera's information gain.") : "");
  }
  function badge(color, label, value, title, ring) {
    return "<span class='badge' title='" + esc(title || "") + "'><i style='background:" + color + "'></i><span>" + esc(label) + "</span> " + esc(value) +
      (ring ? "<b class='tmark' style='border-color:" + ring + "'></b>" : "") + "</span>";
  }
  function energyBar(sl) {
    var nb = sl.energy_belief.length, sw = 50, gap = 3, W = nb * sw + (nb - 1) * gap, H = 40, c = css("--c4"), s = "";
    s += "<svg class='ebar' viewBox='0 0 " + W + " " + H + "' role='img' aria-label='Energy belief by band; the triangle marks the true band'>";
    for (var i = 0; i < nb; i++) {
      var x = i * (sw + gap), pr = sl.energy_belief[i];
      s += "<g><title>" + esc(nm("energy", i)) + ": belief " + pr.toFixed(2) + (i === sl.energy_true ? " (true band)" : "") + "</title>" +
        "<text x='" + (x + sw / 2) + "' y='10' text-anchor='middle' font-size='10.5'>" + esc(nm("energy", i)) + "</text>" +
        "<rect x='" + x + "' y='14' width='" + sw + "' height='14' rx='2' fill='" + c + "' fill-opacity='" + pr.toFixed(3) + "'/>" +
        "<rect x='" + x + "' y='14' width='" + sw + "' height='14' rx='2' fill='none' stroke='" + css("--line") + "' stroke-width='1'/></g>";
      if (i === sl.energy_true) {
        var cx = x + sw / 2;
        s += "<path d='M" + cx + ",30 l-5,8 h10 z' fill='" + css("--fg") + "'><title>true band: " + esc(nm("energy", i)) + "</title></path>";
      }
    }
    return s + "</svg>";
  }
  function drawBadges(t) {
    var sl = t.slow, h = "<div class='mslabel'>Mission state</div><div class='msbody'>";
    h += "<div class='msg'><span class='msl'>energy (triangle: true)</span>" + energyBar(sl) + "</div>";
    h += "<div class='msg'><span class='msl'>rocks (ring: true quality)</span><span class='msrow'>";
    sl.rocks_belief.forEach(function (b, k) {
      var q = argmax(b);
      h += badge(qualColor(q), "rock " + (k + 1), nm("rock", q) + " " + b[q].toFixed(2), "believed quality; true: " + nm("rock", sl.rocks_true[k]), qualColor(sl.rocks_true[k]));
    });
    h += "</span></div>";
    if (sl.zone != null) h += "<div class='msg'><span class='msl'>zone</span><span class='msrow'>" + badge(css("--muted"), "in", pretty(nm("zone", sl.zone)), "the zone the rover is in") + "</span></div>";
    h += "<div class='msg'><span class='msl'>option and sensor</span><span class='msrow'>" +
      badge(optColor(t.option), "option", optLabel(t.option) + (dur(t.option) ? " (" + dur(t.option) + " ticks)" : "")) +
      badge(t.sensor ? css("--c3") : css("--muted"), "sensor", nm("sensor", t.sensor)) + "</span></div></div>";
    $("badges").innerHTML = h;
  }

  // ---- timeline ----
  function drawTimeline() {
    var host = $("timeline"), T = run().ticks, n = T.length, C = cache();
    var W = Math.max(300, host.clientWidth - 16), L = W < 480 ? 50 : 64, R = 8, sx = function (i) { return L + i / n * (W - L - R); };
    var yO = 6, hO = 28, yE = yO + hO + 14, hE = 56, yR = yE + hE + 14, hR = 13, gR = 4, yX = yR + 3 * (hR + gR) + 10, hX = 20, H = yX + hX + 24;
    host.innerHTML = "";
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Timeline of the run", width: W, height: H }, host);
    svg.style.height = "auto";
    function lab(txt, y) { el("text", { x: 4, y: y, "font-size": 10.5, class: "muted" }, svg, txt); }
    // option bands
    lab("option", yO + 18);
    C.segs.forEach(function (s) {
      var x = sx(s.t0), w = sx(s.t1) - x, g = el("g", {}, svg), o = s.opt;
      el("rect", { x: x + 0.5, y: yO, width: Math.max(1, w - 1), height: hO, rx: 3, fill: optColor(o), "fill-opacity": optAlpha(o), stroke: optColor(o), "stroke-width": 1 }, g);
      var d = dur(o);
      g.appendChild(document.createElementNS(NS, "title")).textContent = optLabel(o) + ", ticks " + s.t0 + " to " + (s.t1 - 1) + (d ? ", nominal duration " + d : "");
      if (w > 20) { var tx = el("text", { x: x + w / 2, y: yO + 18, "text-anchor": "middle", "font-size": w > 56 ? 10.5 : 9.5, "paint-order": "stroke", stroke: css("--panel"), "stroke-width": 3 }, g, w > 56 ? optLabel(o) : nm("option", o).replace(/^(\w)\w*?_?(\d?)$/, "$1$2"));
        tx.setAttribute("stroke-linejoin", "round"); }
    });
    // energy
    if (W < 480) el("text", { x: 4, y: yE - 3, "font-size": 10.5, class: "muted" }, svg, "energy"); else lab("energy", yE + 12);
    var nb = names("energy").length || 4, ey = function (b) { return yE + hE - 4 - b / (nb - 1) * (hE - 8); };
    for (var b = 0; b < nb; b++) {
      el("line", { x1: L, x2: W - R, y1: ey(b), y2: ey(b), stroke: css("--line"), "stroke-width": 0.8 }, svg);
      el("text", { x: L - 4, y: ey(b) + 3.5, "text-anchor": "end", "font-size": 9.5, class: "muted" }, svg, nm("energy", b));
    }
    function step(vals, color, dash, w) {
      var d = "";
      vals.forEach(function (v, i) { d += (i ? "L" : "M") + sx(i) + "," + ey(v) + "L" + sx(i + 1) + "," + ey(v); });
      el("path", { d: d, fill: "none", stroke: color, "stroke-width": w, "stroke-dasharray": dash, "stroke-linejoin": "round" }, svg);
    }
    step(T.map(function (t) { return t.slow.energy_true; }), css("--fg"), "", 2);
    step(T.map(function (t) { return argmax(t.slow.energy_belief); }), css("--c4"), "5 3", 2);
    // rocks
    var qs = names("rock").length ? names("rock") : ["good", "bad", "sampled"];
    for (var k = 0; k < 3; k++) {
      var yy = yR + k * (hR + gR);
      lab("rock " + (k + 1), yy + 10);
      T.forEach(function (t, i) {
        var bb = t.slow.rocks_belief[k], yc = yy, w = sx(i + 1) - sx(i);
        for (var q = 0; q < bb.length; q++) {
          var hh = bb[q] * hR; if (hh < 0.2) continue;
          el("rect", { x: sx(i), y: yc, width: w + 0.4, height: hh, fill: qualColor(q) }, svg); yc += hh;
        }
      });
    }
    // exposure
    lab("exposure", yX + 14);
    el("line", { x1: L, x2: W - R, y1: yX + hX, y2: yX + hX, stroke: css("--line") }, svg);
    T.forEach(function (t, i) {
      var h = t.sensor ? hX : hX * 0.4, x = (sx(i) + sx(i + 1)) / 2;
      el("line", { x1: x, x2: x, y1: yX + hX, y2: yX + hX - h, stroke: css(t.sensor ? "--c3" : "--muted"), "stroke-width": Math.max(1, (sx(1) - sx(0)) * 0.55) }, svg);
    });
    // axis
    var stp = n > 90 ? 20 : 10;
    for (var i = 0; i <= n; i += stp) el("text", { x: sx(i), y: H - 6, "text-anchor": "middle", "font-size": 10, class: "muted" }, svg, i);
    // cursor
    var cx = (sx(S.t) + sx(S.t + 1)) / 2;
    el("line", { x1: cx, x2: cx, y1: 0, y2: yX + hX + 4, stroke: css("--accent"), "stroke-width": 2 }, svg);
    var tl = $("tlegend"); tl.innerHTML = "";
    names("option").forEach(function (o, i) {
      tl.innerHTML += "<span><i style='background:" + optColor(i) + ";opacity:" + optAlpha(i) + "'></i>" + esc(pretty(o)) + "</span>";
    });
    tl.innerHTML += "<span>energy: solid true, dashed believed</span>";
    qs.forEach(function (q, i) { tl.innerHTML += "<span><i style='background:" + qualColor(i) + "'></i>rock " + esc(q) + "</span>"; });
    function pick(ev) {
      var r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W;
      S.t = Math.max(0, Math.min(n - 1, Math.floor((x - L) / (W - L - R) * n))); render();
    }
    svg.addEventListener("pointerdown", function (e) { svg.setPointerCapture(e.pointerId); pick(e); svg._d = true; });
    svg.addEventListener("pointermove", function (e) { if (svg._d) pick(e); });
    svg.addEventListener("pointerup", function () { svg._d = false; });
    svg.addEventListener("pointercancel", function () { svg._d = false; });
  }

  // ---- objective ----
  var tipEl;
  function leafRows(p) {
    var out = [];
    Object.keys(p.terms).forEach(function (tm) {
      Object.keys(p.terms[tm]).forEach(function (lf) { var iv = p.terms[tm][lf]; out.push([tm, lf, iv[0], iv[1]]); });
    });
    return out;
  }
  function termSum(p, tm) {
    var lo = 0, hi = 0, d = p.terms[tm] || {};
    Object.keys(d).forEach(function (k) { lo += d[k][0]; hi += d[k][1]; });
    return [lo, hi];
  }
  function ivs(lo, hi) { return lo === hi || Math.abs(hi - lo) < 1e-9 ? fmt(lo).replace("+", "") : "[" + lo.toFixed(2) + ", " + hi.toFixed(2) + "]"; }
  function approx(tm, lf) { var a = S.tier === "rover" && tick().approximate; return !!(a && a[tm] && a[tm][lf]); }
  function anyApprox() { var a = S.tier === "rover" && tick().approximate; if (!a) return false; for (var tm in a) for (var lf in a[tm]) if (a[tm][lf]) return true; return false; }
  function showTip(ev, it) {
    var h = "<b>" + esc(it.name) + "</b> &nbsp; posterior weight qπ=" + it.q.toFixed(3) + "<br>G " + (it.G[0] === it.G[1] ? "= " + it.G[0].toFixed(2) + " (point)" : "[" + it.G[0].toFixed(2) + ", " + it.G[1].toFixed(2) + "]") + "<table>", ap = false;
    leafRows(it.p).forEach(function (r) {
      var a = approx(r[0], r[1]); ap = ap || a;
      h += "<tr><td style='color:" + termColor(r[0]) + "'>" + r[0] + "</td><td>" + esc(r[1]) + (a ? " <span class='ap'>≈</span>" : "") + "</td><td>" + (r[2] === r[3] ? r[2].toFixed(2) : "[" + r[2].toFixed(2) + ", " + r[3].toFixed(2) + "]") + "</td></tr>";
    });
    tipEl.innerHTML = h + "</table>" + (ap ? "<span class='ap'>≈ approximate: read through the decoder's linearisation at the belief</span>" : ""); tipEl.hidden = false;
    var box = ev.currentTarget.getBoundingClientRect(), mouse = ev.clientX != null && ev.type !== "focus";
    var x = mouse ? ev.clientX + 14 : box.left + 40, y = mouse ? ev.clientY + 14 : box.bottom;
    x = Math.min(x, window.innerWidth - tipEl.offsetWidth - 8); y = Math.min(y, window.innerHeight - tipEl.offsetHeight - 8);
    tipEl.style.left = Math.max(4, x) + "px"; tipEl.style.top = Math.max(4, y) + "px";
  }
  function objItems() {
    if (S.tier === "rover") {
      var t = tick();
      return {
        terms: Object.keys(t.policies[0].terms), decided: null,
        items: t.policies.map(function (p, i) { return { name: "π" + p.id, sub: "q=" + p.q_pi.toFixed(2), G: p.G, q: p.q_pi, p: p, chosen: i === t.chosen }; })
      };
    }
    var d = decTick(); if (!d) return { terms: [], items: [], decided: null };
    var items = d.slow_policies.map(function (p, i) {
      return { name: p.options.map(optLabel).join(" → "), sub: "q=" + p.q_pi.toFixed(2), G: p.G, q: p.q_pi, p: p, chosen: i === d.slow_chosen };
    });
    items.sort(function (a, b) { return (a.G[0] + a.G[1]) - (b.G[0] + b.G[1]); });
    items.forEach(function (it, i) { it.rank = i + 1; });
    return { terms: Object.keys(d.slow_policies[0].terms), decided: d.t, items: items };
  }
  function objRange(terms) {
    var C = cache(), key = S.tier, rover = S.tier === "rover";
    if (!rover && C.range[key]) return C.range[key];
    var lo = Infinity, hi = -Infinity;
    (rover ? [tick()] : run().ticks).forEach(function (tk) {
      var ps = rover ? tk.policies : (tk.slow_policies || []);
      ps.forEach(function (p) {
        lo = Math.min(lo, p.G[0]); hi = Math.max(hi, p.G[1]);
        terms.forEach(function (tm) { var s = termSum(p, tm); lo = Math.min(lo, s[0]); hi = Math.max(hi, s[1]); });
      });
    });
    return rover ? [lo, hi] : (C.range[key] = [lo, hi]);
  }
  function stackRange(items, terms) {
    var lo = 0, hi = 0;
    items.forEach(function (it) {
      var neg = 0, pos = 0;
      terms.forEach(function (tm) {
        if (!it.p.terms[tm]) return;
        var s = termSum(it.p, tm), md = (s[0] + s[1]) / 2, hw = (s[1] - s[0]) / 2, c;
        if (md < 0) { neg += md; c = neg - md / 2; } else { pos += md; c = pos - md / 2; }
        lo = Math.min(lo, neg, c - hw); hi = Math.max(hi, pos, c + hw);
      });
    });
    return [lo, hi];
  }
  function drawObjective() {
    var host = $("objective-chart"), mode = $("objmode").value, mission = S.tier === "mission";
    var o = objItems(), items = o.items, terms = o.terms;
    $("objnote").textContent = mission
      ? "Mission tier: the slow model is tabular and its option sequences are enumerated exactly, so each G is a point (lower equals upper) and is drawn as a tick, not a bar."
      : "Rover tier: each bar is the interval the closed-form mixture bounds put around G; the axis is scaled to this tick's policies. The entropy cost over position (uncertainty) is on in every setting; the epistemic switch controls the information-gain leaves, for the modalities the setting names." + (anyApprox() ? " Leaves that read the camera are computed on the decoder's linearisation at the belief and are marked ≈ in the per-leaf view; their brackets are brackets of that linearised model." : "");
    $("objcap").textContent = (mission
      ? "Expected free energy of the kept option sequences (the " + o.items.length + " best of the enumeration) at the most recent decision (tick " + o.decided + "), best first; lower is better. "
      : "Expected free energy G for the current tick (lower is better). ") +
      "Hover or focus a policy for its per-term, per-leaf values. The chosen " + (mission ? "sequence" : "policy") + " is highlighted. " + (B.meta.sign_convention ? "Sign convention: " + B.meta.sign_convention + "." : "") +
      (mode === "terms" ? " Each policy's G is split into its objective terms, stacked from zero: negative terms to the left, positive to the right; a bracket marks each term's interval." : "");
    var btn = $("objall"); btn.hidden = !(mission && items.length > TOP_N);
    btn.textContent = S.showAll ? "Show top " + TOP_N : "Show all " + items.length + " kept";
    if (mission && !S.showAll && items.length > TOP_N) {
      var top = items.slice(0, TOP_N), ch = items.filter(function (i) { return i.chosen; })[0];
      if (ch && top.indexOf(ch) < 0) top.push(ch);
      items = top;
    }
    host.innerHTML = "";
    if (!items.length) { host.textContent = "No decision yet."; return; }
    var rg = mode === "terms" ? stackRange(items, terms) : objRange(terms), lo = Math.min(rg[0], 0), hi = rg[1], padx = (hi - lo) * 0.03; lo -= padx; hi += padx;
    var W = Math.max(300, host.clientWidth - 16), L = mission ? (W < 520 ? 124 : 190) : 74, R = 14, rowH = mode === "total" ? 36 : 58, top0 = 8, H = top0 + items.length * rowH + 34;
    var sx = function (v) { return L + (v - lo) / (hi - lo) * (W - L - R); };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": "Expected free energy per " + (mission ? "option sequence" : "policy") }, host);
    svg.style.height = "auto";
    var step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 5))); step *= (hi - lo) / step / 5 > 3 ? 5 : (hi - lo) / step / 5 > 1.5 ? 2 : 1;
    for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      el("line", { x1: sx(v), x2: sx(v), y1: top0, y2: H - 26, stroke: css("--line"), "stroke-width": v === 0 ? 1.6 : 0.8 }, svg);
      el("text", { x: sx(v), y: H - 10, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg, (Math.round(v * 100) / 100) + "");
    }
    function mark(g, iv, y, h, color, op, strong) {
      var w = sx(iv[1]) - sx(iv[0]);
      if (w < 1.5) {   // point value: a tick
        el("line", { x1: sx(iv[0]), x2: sx(iv[0]), y1: y, y2: y + h, stroke: color, "stroke-opacity": op, "stroke-width": strong ? 4 : 3, "stroke-linecap": "round" }, g);
      } else {
        el("rect", { x: sx(iv[0]), y: y, width: w, height: h, rx: 2, fill: color, "fill-opacity": op }, g);
        var m = (iv[0] + iv[1]) / 2; el("line", { x1: sx(m), x2: sx(m), y1: y - 2, y2: y + h + 2, stroke: css("--fg"), "stroke-width": 2 }, g);
      }
    }
    items.forEach(function (it, i) {
      var y = top0 + i * rowH, ch = it.chosen, p = it.p;
      var g = el("g", { class: "row", tabindex: 0, "aria-label": it.name + ", G " + it.G[0].toFixed(2) + (it.G[0] === it.G[1] ? "" : " to " + it.G[1].toFixed(2)) + (ch ? ", chosen" : "") }, svg);
      el("rect", { class: "rowbg", x: 0, y: y, width: W, height: rowH, fill: "transparent" }, g);
      if (ch) el("rect", { x: 1, y: y + 1, width: W - 2, height: rowH - 2, fill: "none", stroke: css("--accent"), "stroke-width": 1.5, rx: 3 }, g);
      var label = (ch ? "▸ " : "") + (mission ? "#" + it.rank + " " : "") + it.name;
      var maxc = Math.floor((L - 12) / 6.2);
      el("text", { x: 8, y: y + rowH / 2 - 1, "font-size": 11.5, "font-weight": ch ? 700 : 400 }, g, label.length > maxc ? label.slice(0, maxc - 1) + "…" : label);
      el("text", { x: 8, y: y + rowH / 2 + 14, "font-size": 9.5, class: "muted" }, g, it.sub);
      if (mode === "total") {
        mark(g, it.G, y + 8, rowH - 16, css(ch ? "--accent" : "--muted"), ch ? 0.9 : 0.55, ch);
        if (it.G[1] - it.G[0] >= 1e-9 && sx(it.G[1]) - sx(it.G[0]) >= 1.5) { /* midpoint drawn in mark */ }
      } else {
        var ts = terms.filter(function (tm) { return p.terms[tm]; }), neg = 0, pos = 0, by = y + 18, bhh = 20;
        ts.forEach(function (tm) {
          var s = termSum(p, tm), md = (s[0] + s[1]) / 2, from, to;
          if (md < 0) { from = neg; neg += md; to = neg; } else { from = pos; pos += md; to = pos; }
          var x0 = Math.min(sx(from), sx(to)), w = Math.abs(sx(to) - sx(from)), cx = (sx(from) + sx(to)) / 2, hw = (sx(s[1]) - sx(s[0])) / 2;
          if (w > 0.3) el("rect", { x: x0, y: by, width: w, height: bhh, fill: termColor(tm), "fill-opacity": 0.8, stroke: css("--panel"), "stroke-width": 0.6 }, g);
          if (hw >= 1.2) {
            el("path", { d: "M" + (cx - hw) + "," + (by - 7) + "v5 M" + (cx + hw) + "," + (by - 7) + "v5 M" + (cx - hw) + "," + (by - 4.5) + "H" + (cx + hw), fill: "none", stroke: css("--fg"), "stroke-opacity": 0.75, "stroke-width": 1.4 }, g);
          }
        });
        var mg = (it.G[0] + it.G[1]) / 2; el("path", { d: "M" + sx(mg) + "," + (by + bhh + 3) + " l-4,6 h8 z", fill: css("--fg") }, g);
      }
      var show = function (e) { showTip(e, it); }, hide = function () { tipEl.hidden = true; };
      g.addEventListener("mousemove", show); g.addEventListener("mouseleave", hide);
      g.addEventListener("focus", show); g.addEventListener("blur", hide);
    });
    var lg = $("objlegend"); lg.innerHTML = "";
    if (mode === "terms") {
      terms.forEach(function (tm) {
        var lv = {};
        items.forEach(function (it) { Object.keys(it.p.terms[tm] || {}).forEach(function (k) { lv[k] = 1; }); });
        lg.innerHTML += "<span><i style='background:" + termColor(tm) + "'></i>" + esc(tm) + " (" + esc(Object.keys(lv).join(", ")) + ")</span>";
      });
      lg.innerHTML += "<span>bracket: term interval</span><span>&#9650; total G midpoint</span>";
    }
    else if (mission) lg.innerHTML = "<span><i style='background:" + css("--muted") + "'></i>G (point)</span><span><i style='background:" + css("--accent") + "'></i>chosen</span>";
    else lg.innerHTML = "<span><i style='background:" + css("--muted") + "'></i>interval</span><span>| midpoint</span><span><i style='background:" + css("--accent") + "'></i>chosen</span>";
    if (mode === "terms" && anyApprox()) lg.innerHTML += "<span class='approx'>≈ approximate leaves (hover)</span>";
  }

  // ---- planner ----
  function drawPlanner() {
    var c = $("plan"), m = setupCanvas(c), g = m.g, t = tick();
    drawGrid(m); drawZones(m, t);
    var dyn = B.meta.dynamics, terr = argmax(t.belief.terrain), v0 = t.belief.velocity || [0, 0];
    var longIdx = Math.max(0, names("sensor").findIndex(function (s) { return /long/i.test(s); }));
    function path(p, color, wLong, wShort, alpha) {
      var x = p.start[0], y = p.start[1], vx = v0[0], vy = v0[1];
      g.globalAlpha = alpha; g.strokeStyle = color; g.fillStyle = color; g.lineCap = "round";
      p.v_plan.forEach(function (a, k) {
        var lg = p.sensor_plan[k] === longIdx, lv = p.sensor_plan[k] ? 1 : 0, drift = dyn.drift[terr], x0 = x, y0 = y;
        x += dyn.dt * vx; y += dyn.dt * vy;
        vx = dyn.keep[lv] * vx + dyn.gain[lv] * a[0] + drift[0]; vy = dyn.keep[lv] * vy + dyn.gain[lv] * a[1] + drift[1];
        g.lineWidth = lg ? wLong : wShort; g.beginPath(); g.moveTo(m.X(x0), m.Y(y0)); g.lineTo(m.X(x), m.Y(y)); g.stroke();
        if (lg) { g.beginPath(); g.arc(m.X(x), m.Y(y), wLong * 0.85 + 1, 0, 2 * Math.PI); g.fill(); }
      });
      g.globalAlpha = 1;
    }
    var order = t.policies.map(function (p, i) { return i; }).sort(function (a, b) { return t.policies[b].q_pi - t.policies[a].q_pi; }).slice(0, 4);
    if (order.indexOf(t.chosen) < 0) order.push(t.chosen);
    order.forEach(function (i) { if (i !== t.chosen) path(t.policies[i], css("--muted"), 2.8, 1, 0.6); });
    var ch = t.policies[t.chosen]; path(ch, css("--accent"), 5, 1.6, 1);
    g.beginPath(); g.arc(m.X(ch.start[0]), m.Y(ch.start[1]), 6, 0, 2 * Math.PI); g.fillStyle = css("--fg"); g.fill();
    var lw = function (c, w, dot) { return "<svg class='lsw' viewBox='0 0 28 10'><line x1='2' x2='" + (dot ? 21 : 26) + "' y1='5' y2='5' stroke='" + c + "' stroke-width='" + w + "' stroke-linecap='round'/>" + (dot ? "<circle cx='23' cy='5' r='3.5' fill='" + c + "'/>" : "") + "</svg>"; };
    $("planlegend").innerHTML = "<span>" + lw(css("--fg"), 4, true) + esc(nm("sensor", longIdx)) + " step: thick, dot at its end</span><span>" +
      lw(css("--fg"), 1.2, false) + "short exposure step: thin</span><span><i style='background:" + css("--accent") + "'></i>chosen policy</span><span><i style='background:" + css("--muted") + "'></i>other top policies</span>";
    // mission plan
    var d = decTick(), ul = $("missionplan"); ul.innerHTML = "";
    if (!d) { $("missionnote").textContent = ""; return; }
    var sp = d.slow_policies[d.slow_chosen];
    sp.options.forEach(function (o, i) {
      var li = document.createElement("li"); li.style.borderLeftColor = optColor(o);
      if (i === 0) li.className = "now";
      li.innerHTML = (i + 1) + ". <b>" + esc(optLabel(o)) + "</b><small>" + (dur(o) != null ? dur(o) + " ticks" : "") + "</small>";
      ul.appendChild(li);
    });
    $("missionnote").textContent = "Chosen at the decision on tick " + d.t + " (G " + sp.G[0].toFixed(2) + ", qπ " + sp.q_pi.toFixed(2) + "); the outlined option is the one executing, the rest is the plan beyond it.";
  }

  // ---- mdl ----
  function drawMdl() {
    var tb = document.querySelector("#mdltable tbody"), tf = document.querySelector("#mdltable tfoot");
    tb.innerHTML = ""; tf.innerHTML = "";
    var order = ["model", "mission", "rover"], by = {}, mx = 0, tot = 0;
    B.mdl.components.forEach(function (c) { (by[c.tier] = by[c.tier] || []).push(c); if (c.nats != null) { mx = Math.max(mx, Math.log10(c.nats + 1)); tot += c.nats; } });
    Object.keys(by).forEach(function (k) { if (order.indexOf(k) < 0) order.push(k); });
    function comp(c) {
      var tr = document.createElement("tr"), un = c.nats == null;
      var a = document.createElement("td"); a.textContent = short(c.name.replace(/^[a-z_]+:/, "")); a.title = c.name; tr.appendChild(a);
      var k = document.createElement("td"); k.className = "kind"; k.textContent = pretty(c.kind); tr.appendChild(k);
      var pc = document.createElement("td"); pc.className = "num muted"; pc.textContent = c.entries != null ? c.entries.toLocaleString() : ""; pc.title = c.entries != null ? "array entries the params hold at this address" : ""; tr.appendChild(pc);
      var n = document.createElement("td"); n.className = "num" + (un ? " unscored" : ""); n.textContent = un ? "unscored" : c.nats.toFixed(2); tr.appendChild(n);
      var bc = document.createElement("td"); bc.className = "mdlbarcell";
      if (!un) { var bar = document.createElement("span"); bar.className = "mdlbar"; bar.style.width = (100 * Math.log10(c.nats + 1) / (mx || 1)).toFixed(1) + "%"; bar.title = "log10(nats + 1) = " + Math.log10(c.nats + 1).toFixed(2); bc.appendChild(bar); }
      tr.appendChild(bc);
      var e = document.createElement("td"); e.textContent = un ? "unscored" : (c.exact ? "exact" : "approximate"); if (!un && !c.exact) e.className = "approx"; tr.appendChild(e);
      tb.appendChild(tr);
    }
    order.forEach(function (tier) {
      var cs = by[tier]; if (!cs) return;
      if (tier !== "model") {
        var sub = 0, ex = 0; cs.forEach(function (c) { if (c.nats != null) { sub += c.nats; if (c.exact) ex += c.nats; } });
        var h = document.createElement("tr"); h.className = "grp light";
        h.innerHTML = "<td colspan='3'>" + esc(tier + " tier subtotal") + "</td><td class='num'>" + sub.toFixed(2) + "</td><td></td><td>" + ex.toFixed(2) + " of it exact</td>";
        tb.appendChild(h);
      }
      cs.forEach(comp);
    });
    $("mdltotal").innerHTML = "Total over scored components: <b>" + Number(tot.toPrecision(4)).toLocaleString() + " nats</b>";
    $("mdlfoot").textContent = "Of this, " + B.mdl.total_exact.toFixed(2) + " nats are exact: the graph and control entries. Every fixed table or density is priced by its family alone and flagged approximate, and its cost grows with its parameter count (entries: the array entries the params hold at that address, a table's whole table or a network's weights). Bars are proportional to log10(nats + 1).";
  }

  // ---- state / wiring ----
  function render() {
    $("tick").value = S.t; $("tickout").textContent = S.t;
    drawField(); updateGraph(); drawTimeline(); drawObjective(); drawPlanner();
  }
  function setPlaying(on) {
    S.playing = on; $("play").textContent = on ? "Pause" : "Play"; $("play").setAttribute("aria-label", on ? "Pause" : "Play");
    clearInterval(S.timer);
    if (on) S.timer = setInterval(function () { S.t = (S.t + 1) % run().ticks.length; render(); }, 450);
  }
  function fullRedraw() { drawGraph(); render(); }
  function hashParams() {
    var o = {}; location.hash.replace(/^#/, "").split("&").forEach(function (kv) { var p = kv.split("="); if (p[0]) o[p[0]] = decodeURIComponent(p[1] || ""); });
    return o;
  }

  function init(b) {
    B = b; tipEl = $("tip"); WIN = computeWindow();
    $("status").hidden = true;
    var hp = hashParams();
    if (hp.theme) document.documentElement.setAttribute("data-theme", hp.theme);
    var vs = $("version"); if (vs && B.meta.library_version) vs.textContent = "v" + B.meta.library_version;
    var sel = $("setting"), groups = {};
    B.settings.forEach(function (s) {
      var gname = s.camera === "encoder" ? "Recogniser replaces the inversion" : s.epistemic ? "Epistemic on" : "Epistemic off";
      if (!groups[gname]) { groups[gname] = document.createElement("optgroup"); groups[gname].label = gname; sel.appendChild(groups[gname]); }
      var o = document.createElement("option"); o.value = s.id; o.textContent = s.label; groups[gname].appendChild(o);
    });
    S.setting = B.runs[hp.setting] ? hp.setting : B.settings[0].id; sel.value = S.setting;
    var n = run().ticks.length; $("tick").max = n - 1;
    S.t = Math.max(0, Math.min(n - 1, +hp.t || 0));
    if (hp.tier === "mission") { S.tier = "mission"; $("objtier").value = "mission"; }
    if (hp.view === "terms") $("objmode").value = "terms";
    sel.addEventListener("change", function () { S.setting = sel.value; S.t = $("restart").checked ? 0 : Math.min(S.t, run().ticks.length - 1); $("tick").max = run().ticks.length - 1; CACHE[S.setting] && delete CACHE[S.setting].range.rover; fullRedraw(); });
    $("tick").addEventListener("input", function (e) { S.t = +e.target.value; render(); });
    $("play").addEventListener("click", function () { setPlaying(!S.playing); });
    $("objmode").addEventListener("change", drawObjective);
    $("objtier").addEventListener("change", function (e) { S.tier = e.target.value; drawObjective(); });
    $("objall").addEventListener("click", function () { S.showAll = !S.showAll; drawObjective(); });
    drawMdl(); fullRedraw();
    var rt; window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(render, 100); });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fullRedraw);
    window.__app = { set: function (o) {
      if (o.setting && B.runs[o.setting]) { S.setting = o.setting; $("setting").value = o.setting; drawGraph(); }
      if (o.tier) { S.tier = o.tier; $("objtier").value = o.tier; }
      if (o.view) $("objmode").value = o.view;
      if (o.all != null) S.showAll = o.all;
      if (o.t != null) S.t = o.t; render();
    } };
    window.__ready = true;
  }
  function start() {
    load().then(init).catch(function (e) {
      $("status").hidden = true;
      var d = document.createElement("p"); d.className = "err"; d.textContent = "Could not load the replay data: " + e.message;
      document.querySelector("main").prepend(d);
    });
  }
  // lazy: let the page paint first, then pull the (large) bundle
  requestAnimationFrame(function () { setTimeout(start, 0); });
})();
