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
  function drawGraph() {
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
      halo(g, "beacon", bx, by + 22);
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
      (t.free_energy != null ? "<span>free energy: <b>" + t.free_energy.toFixed(2) + "</b></span>" : "") +
      (t.seconds != null ? "<span>tick time: <b>" + t.seconds.toFixed(1) + " s</b></span>" : "");
    drawCamera(t);
  }
  function pixels(c, rows, overlay) {
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
  function imgXY(sz) {
    var f = B.meta.field, sx = sz / (f[0][1] - f[0][0]), sy = sz / (f[1][1] - f[1][0]);
    return { X: function (x) { return (x - f[0][0]) * sx; }, Y: function (y) { return sz - (y - f[1][0]) * sy; }, sx: sx, sy: sy };
  }
  function drawCamera(t) {
    var host = $("camgrid");
    if (!t.img) { $("camerafig").hidden = true; return; }
    $("camerafig").hidden = false;
    if (!host.children.length) {
      ["observed image", "decoder prediction at the belief", "where the image says the rover is"].forEach(function (lbl, i) {
        var d = document.createElement("div"); d.className = "cam";
        d.innerHTML = "<canvas id='cam" + i + "' width='160' height='160' role='img' aria-label='" + lbl + "'></canvas><span class='cl'>" + lbl + "</span><span class='cv' id='camv" + i + "'></span>";
        host.appendChild(d);
      });
    }
    pixels($("cam0"), t.img);
    pixels($("cam1"), t.recon || t.img);
    var enc = t.encoder, cam = t.camera || {};
    pixels($("cam2"), t.img, function (g, sz) {
      var m = imgXY(sz), bx = m.X(t.true_x[0]), by = m.Y(t.true_x[1]);
      t.belief.components.forEach(function (cm) {
        var rx = 2 * Math.sqrt(Math.max(cm.cov[0][0], 0)) * m.sx, ry = 2 * Math.sqrt(Math.max(cm.cov[1][1], 0)) * m.sy;
        g.beginPath(); g.ellipse(m.X(cm.mu[0]), m.Y(cm.mu[1]), rx, ry, 0, 0, 2 * Math.PI);
        g.strokeStyle = css("--c1"); g.globalAlpha = Math.min(1, cm.w + 0.3); g.lineWidth = 1.5; g.stroke(); g.globalAlpha = 1;
      });
      if (enc) {
        g.beginPath(); g.ellipse(m.X(enc.mean[0]), m.Y(enc.mean[1]), 2 * enc.sd[0] * m.sx, 2 * enc.sd[1] * m.sy, 0, 0, 2 * Math.PI);
        g.setLineDash([3, 3]); g.strokeStyle = css("--c4"); g.lineWidth = 1.5; g.stroke(); g.setLineDash([]);
      }
      g.beginPath(); g.arc(bx, by, 3.5, 0, 2 * Math.PI); g.fillStyle = css("--fg"); g.fill(); g.strokeStyle = css("--bg"); g.lineWidth = 1.5; g.stroke();
    });
    $("camv1").textContent = cam.log_evidence != null ? "log evidence " + cam.log_evidence.toFixed(1) :
      cam.covered_term != null ? "covered term " + cam.covered_term.toFixed(1) + ", normaliser " + cam.normaliser.toFixed(1) : "";
    $("camv2").textContent = enc ? "encoder " + enc.mean[0].toFixed(2) + ", " + enc.mean[1].toFixed(2) + " ± " + enc.sd[0].toFixed(2) + ", " + enc.sd[1].toFixed(2) + (encoderSetting() ? " (used)" : " (diagnostic)") : "";
    $("camcap").innerHTML = "The camera: the observed 8&times;8 image (left), the decoder's prediction at the belief mean and most likely terrain (middle), and the same image with the belief's components (solid), the encoder's measurement (dashed) and the true position (dot) over it (right). " +
      (encoderSetting() ? "In this setting the encoder's measurement replaces the inversion of the image; the covered term is the image's expected cost under the belief, and the normaliser is the measurement's evidence." :
        "In this setting the engine inverts the decoder at the belief; the number under the middle panel is the image's log evidence under that linearisation. The encoder is run for comparison only.");
  }
  function badge(color, label, value, title) {
    return "<span class='badge' title='" + esc(title || "") + "'><i style='background:" + color + "'></i><span>" + esc(label) + "</span> " + esc(value) + "</span>";
  }
  function drawBadges(t) {
    var sl = t.slow, h = "";
    h += badge(optColor(t.option), "option", optLabel(t.option) + (dur(t.option) ? " (" + dur(t.option) + " ticks)" : ""));
    h += badge(t.sensor ? css("--c3") : css("--muted"), "sensor", nm("sensor", t.sensor));
    var eb = argmax(sl.energy_belief);
    h += badge(css("--c4"), "energy", nm("energy", eb) + " " + sl.energy_belief[eb].toFixed(2), "believed band; true band: " + nm("energy", sl.energy_true));
    sl.rocks_belief.forEach(function (b, k) {
      var q = argmax(b);
      h += badge(qualColor(q), "rock " + (k + 1), nm("rock", q) + " " + b[q].toFixed(2), "believed quality; true: " + nm("rock", sl.rocks_true[k]));
    });
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
    lab("energy", yE + 12);
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
    var h = "<b>" + esc(it.name) + "</b> &nbsp; qπ=" + it.q.toFixed(3) + "<br>G " + (it.G[0] === it.G[1] ? "= " + it.G[0].toFixed(2) + " (point)" : "[" + it.G[0].toFixed(2) + ", " + it.G[1].toFixed(2) + "]") + "<table>", ap = false;
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
  function drawObjective() {
    var host = $("objective-chart"), mode = $("objmode").value, mission = S.tier === "mission";
    var o = objItems(), items = o.items, terms = o.terms;
    $("objnote").textContent = mission
      ? "Mission tier: the slow model is tabular and its option sequences are enumerated exactly, so each G is a point (lower equals upper) and is drawn as a tick, not a bar."
      : "Rover tier: each bar is the interval the closed-form mixture bounds put around G; the axis is scaled to this tick's policies. The entropy cost over position (uncertainty) is on in every setting; the epistemic switch controls the information-gain leaves." + (anyApprox() ? " Leaves that read the camera are computed on the decoder's linearisation at the belief and are marked ≈ in the per-leaf view; their brackets are brackets of that linearised model." : "");
    $("objcap").textContent = (mission
      ? "Expected free energy of each option sequence at the most recent decision (tick " + o.decided + "), best first; lower is better. "
      : "Expected free energy G for the current tick (lower is better). ") +
      "Hover or focus a policy for its per-term, per-leaf values. The chosen " + (mission ? "sequence" : "policy") + " is highlighted. " + (B.meta.sign_convention ? "Sign convention: " + B.meta.sign_convention + "." : "");
    var btn = $("objall"); btn.hidden = !(mission && items.length > TOP_N);
    btn.textContent = S.showAll ? "Show top " + TOP_N : "Show all " + items.length;
    if (mission && !S.showAll && items.length > TOP_N) {
      var top = items.slice(0, TOP_N), ch = items.filter(function (i) { return i.chosen; })[0];
      if (ch && top.indexOf(ch) < 0) top.push(ch);
      items = top;
    }
    host.innerHTML = "";
    if (!items.length) { host.textContent = "No decision yet."; return; }
    var rg = objRange(terms), lo = Math.min(rg[0], 0), hi = rg[1], padx = (hi - lo) * 0.03; lo -= padx; hi += padx;
    var W = Math.max(300, host.clientWidth - 16), L = mission ? (W < 520 ? 124 : 190) : 74, R = 14, rowH = mode === "total" ? 36 : 54, top0 = 8, H = top0 + items.length * rowH + 34;
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
        var ts = terms.filter(function (tm) { return p.terms[tm]; }), bh = (rowH - 10) / ts.length;
        ts.forEach(function (tm, j) { mark(g, termSum(p, tm), y + 5 + j * bh + 1, bh - 2, termColor(tm), 0.85, false); });
        var mg = (it.G[0] + it.G[1]) / 2; el("path", { d: "M" + sx(mg) + "," + (y + rowH - 3) + " l-4,5 h8 z", fill: css("--fg") }, g);
      }
      var show = function (e) { showTip(e, it); }, hide = function () { tipEl.hidden = true; };
      g.addEventListener("mousemove", show); g.addEventListener("mouseleave", hide);
      g.addEventListener("focus", show); g.addEventListener("blur", hide);
    });
    var lg = $("objlegend"); lg.innerHTML = "";
    if (mode === "terms") terms.forEach(function (tm) { lg.innerHTML += "<span><i style='background:" + termColor(tm) + "'></i>" + tm + "</span>"; });
    else if (mission) lg.innerHTML = "<span><i style='background:" + css("--muted") + "'></i>G (point)</span><span><i style='background:" + css("--accent") + "'></i>chosen</span>";
    else lg.innerHTML = "<span><i style='background:" + css("--muted") + "'></i>interval</span><span>| midpoint</span><span><i style='background:" + css("--accent") + "'></i>chosen</span>";
    if (mode === "terms" && anyApprox()) lg.innerHTML += "<span class='approx'>≈ approximate leaves (hover)</span>";
  }

  // ---- planner ----
  function drawPlanner() {
    var c = $("plan"), m = setupCanvas(c), g = m.g, t = tick();
    drawGrid(m); drawZones(m, t);
    var dyn = B.meta.dynamics, terr = argmax(t.belief.terrain), v0 = t.belief.velocity || [0, 0];
    function path(p, color, w, alpha) {
      var x = p.start[0], y = p.start[1], vx = v0[0], vy = v0[1], pts = [[x, y]];
      p.v_plan.forEach(function (a, k) {
        var lv = p.sensor_plan[k] ? 1 : 0, drift = dyn.drift[terr];
        x += dyn.dt * vx; y += dyn.dt * vy;
        vx = dyn.keep[lv] * vx + dyn.gain[lv] * a[0] + drift[0]; vy = dyn.keep[lv] * vy + dyn.gain[lv] * a[1] + drift[1];
        pts.push([x, y]);
      });
      g.globalAlpha = alpha; g.strokeStyle = color; g.lineWidth = w; g.lineJoin = "round"; g.beginPath();
      pts.forEach(function (q, i) { i ? g.lineTo(m.X(q[0]), m.Y(q[1])) : g.moveTo(m.X(q[0]), m.Y(q[1])); }); g.stroke();
      pts.slice(1).forEach(function (q, i) {
        var long = p.sensor_plan[i] ? 1 : 0, r = (long ? 4.2 : 3.6) * (w > 2 ? 1.2 : 1);
        g.beginPath(); g.arc(m.X(q[0]), m.Y(q[1]), r, 0, 2 * Math.PI);
        g.fillStyle = long ? color : css("--panel"); g.fill(); g.lineWidth = 1.4; g.stroke();
      });
      g.globalAlpha = 1;
    }
    t.policies.forEach(function (p, i) { if (i !== t.chosen) path(p, css("--muted"), 1.3, 0.5); });
    var ch = t.policies[t.chosen]; path(ch, css("--accent"), 3.2, 1);
    g.beginPath(); g.arc(m.X(ch.start[0]), m.Y(ch.start[1]), 6, 0, 2 * Math.PI); g.fillStyle = css("--fg"); g.fill();
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
    var order = ["model", "mission", "rover"], by = {};
    B.mdl.components.forEach(function (c) { (by[c.tier] = by[c.tier] || []).push(c); });
    Object.keys(by).forEach(function (k) { if (order.indexOf(k) < 0) order.push(k); });
    order.forEach(function (tier) {
      var cs = by[tier]; if (!cs) return;
      var sub = 0, ex = 0; cs.forEach(function (c) { if (c.nats != null) { sub += c.nats; if (c.exact) ex += c.nats; } });
      var h = document.createElement("tr"); h.className = "grp";
      h.innerHTML = "<td colspan='2'>" + esc(tier === "model" ? "whole model" : tier + " tier") + "</td><td class='num'>" + sub.toFixed(2) + "</td><td>" + ex.toFixed(2) + " of it exact</td>";
      tb.appendChild(h);
      cs.forEach(function (c) {
        var tr = document.createElement("tr"), un = c.nats == null;
        var a = document.createElement("td"); a.textContent = short(c.name.replace(/^[a-z_]+:/, "")); a.title = c.name; tr.appendChild(a);
        var k = document.createElement("td"); k.className = "kind"; k.textContent = pretty(c.kind); tr.appendChild(k);
        var n = document.createElement("td"); n.className = "num" + (un ? " unscored" : ""); n.textContent = un ? "unscored" : c.nats.toFixed(2); tr.appendChild(n);
        var e = document.createElement("td"); e.textContent = un ? "unscored" : (c.exact ? "exact" : "approximate"); if (!un && !c.exact) e.className = "approx"; tr.appendChild(e);
        tb.appendChild(tr);
      });
    });
    var m = B.mdl;
    tf.innerHTML = "<tr><td colspan='2'>total (all scored components)</td><td class='num'>" + m.total.toFixed(2) + "</td><td class='approx'>approximate</td></tr>" +
      "<tr><td colspan='2'>exact total (exact components only)</td><td class='num'>" + m.total_exact.toFixed(2) + "</td><td>exact</td></tr>" +
      (m.library_total != null && Math.abs(m.library_total - m.total) > 1e-6 ? "<tr><td colspan='2'>library_total</td><td class='num'>" + m.library_total.toFixed(2) + "</td><td></td></tr>" : "");
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
      var gname = s.camera === "encoder" ? "Encoder measurement" : s.epistemic ? "Epistemic on" : "Epistemic off";
      if (!groups[gname]) { groups[gname] = document.createElement("optgroup"); groups[gname].label = gname; sel.appendChild(groups[gname]); }
      var o = document.createElement("option"); o.value = s.id; o.textContent = s.label; groups[gname].appendChild(o);
    });
    S.setting = B.runs[hp.setting] ? hp.setting : B.settings[0].id; sel.value = S.setting;
    var n = run().ticks.length; $("tick").max = n - 1;
    S.t = Math.max(0, Math.min(n - 1, +hp.t || 0));
    if (hp.tier === "mission") { S.tier = "mission"; $("objtier").value = "mission"; }
    if (hp.view === "terms") $("objmode").value = "terms";
    sel.addEventListener("change", function () { S.setting = sel.value; S.t = Math.min(S.t, run().ticks.length - 1); CACHE[S.setting] && delete CACHE[S.setting].range.rover; fullRedraw(); });
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
