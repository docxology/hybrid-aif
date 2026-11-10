(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var B = null, S = { setting: null, t: 0, playing: false, timer: null, tier: "mission", showAll: false, layer: "terrain" };
  var $ = function (id) { return document.getElementById(id); };
  var TOP_N = 12, CACHE = {}, WIN = null, BG = null, tipEl;

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
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function hexrgb(h) { h = h.replace("#", ""); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }

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
  function other() { for (var k in B.runs) if (k !== S.setting) return k; return S.setting; }
  function settingRec(id) { for (var i = 0; i < B.settings.length; i++) if (B.settings[i].id === (id || S.setting)) return B.settings[i]; return null; }
  function twoLevel() { var r = settingRec(); return !!(r && r.tiers === 2); }
  function cache(id) {
    id = id || S.setting;
    if (CACHE[id]) return CACHE[id];
    var T = B.runs[id].ticks, last = [], li = -1, segs = [];
    T.forEach(function (t, i) {
      if (t.slow && t.slow.decision) li = i;
      last.push(li);
      if (!segs.length || segs[segs.length - 1].option !== t.option) segs.push({ option: t.option, t0: i });
    });
    segs.forEach(function (s, i) { s.t1 = i + 1 < segs.length ? segs[i + 1].t0 : T.length; });
    return (CACHE[id] = { last: last, segs: segs });
  }
  function decTick() { var i = cache().last[S.t]; return i < 0 ? null : run().ticks[i]; }

  // ---- names and colours ----
  var TERM_VAR = { risk: "--c1", uncertainty: "--c2", ambiguity: "--c5", infogain: "--c3", utility: "--c4" };
  function termColor(t) { return css(TERM_VAR[t] || "--muted"); }
  function optColor(i) { return css("--o" + i); }
  function optLabel(i) { return (B.meta.tiers.options || [])[i] || "option " + i; }
  function optShort(i) { var n = optLabel(i), m = /(\d)$/.exec(n); return m ? "S" + m[1] : n === "charge" ? "C" : n; }
  function sectorOf(opt) { var m = /survey sector (\d)/.exec(optLabel(opt)); return m ? +m[1] - 1 : -1; }
  function terrColor(c) { return css(["--mf", "--mr", "--mi"][c]); }
  function terrBg(c) { return css(["--tf", "--tr", "--ti"][c]); }

  // ---- the model graph: both tiers, the parameter variable between them ----
  var POS = {
    "mission.u_option": [140, 52], "mission.energy": [140, 130], "mission.visit": [280, 130], "mission.map": [420, 130],
    "mission.gauge": [140, 208], "mission.cell_obs": [280, 208],
    "rover.field": [420, 282],
    "rover.v": [150, 366], "rover.x": [150, 440], "rover.energy": [285, 440], "rover.terrain": [420, 440],
    "rover.fix": [85, 528], "rover.gauge": [225, 528], "rover.img": [365, 528]
  };
  var BEND = { "rover.x>rover.energy": 0, "rover.terrain>rover.energy": 0, "rover.x>rover.img": -14, "rover.terrain>rover.img": 0, "rover.energy>rover.img": 8 };
  var FAM = { categorical: "categorical", gaussian_mixture: "Gauss. mixture", linear_gaussian: "lin. Gaussian", neural: "neural (MLP)" };
  var KINDS = {
    transition: { dash: "", color: "--fg", w: 1.3, label: "transition" },
    likelihood: { dash: "5 3", color: "--c1", w: 1.3, label: "likelihood" },
    target: { dash: "2 3", color: "--c2", w: 2, label: "preference (target)" },
    parameter: { dash: "", color: "--c3", w: 3, label: "parameter variable read" }
  };
  function drawGraph() {
    var svg = $("graph"); svg.innerHTML = "";
    var W = 520, H = 600;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var defs = el("defs", {}, svg);
    Object.keys(KINDS).forEach(function (k) {
      [["ar-", "auto"], ["as-", "auto-start-reverse"]].forEach(function (p) {
        var m = el("marker", { id: p[0] + k, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: p[1] }, defs);
        el("path", { d: "M0,0 L10,5 L0,10 z", fill: css(KINDS[k].color) }, m);
      });
    });
    // tier frames
    var frames = [[8, 12, 504, 232, "mission tier (slow): the scheduler, one decision every " + B.meta.tiers.slow_ticks + " ticks"], [8, 316, 504, 268, "rover tier (fast): one step every tick"]];
    frames.forEach(function (f) {
      el("rect", { x: f[0], y: f[1], width: f[2], height: f[3], rx: 8, fill: "none", stroke: css("--line"), "stroke-width": 1.5 }, svg);
      el("text", { x: f[0] + 12, y: f[1] + 18, "font-size": 11.5, "font-weight": 600, class: "muted" }, svg, f[4]);
    });
    var info = {};
    B.graph.tiers.forEach(function (t) {
      t.factors.forEach(function (n) { info[n.name] = { kind: "factor", nd: n }; });
      t.modalities.forEach(function (n) { info[n.name] = { kind: "modality", nd: n }; });
      t.controls.forEach(function (n) { info[n.name] = { kind: "control", nd: n }; });
    });
    (B.graph.parameters || []).forEach(function (p) { info[p.name] = { kind: "parameter", nd: p }; });
    // edges (reciprocal transitions drawn once, with a head at each end)
    var edges = B.graph.edges, done = {};
    function pathFor(a, b, bend, r1, r2) {
      var dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
      var x1 = a[0] + ux * r1, y1 = a[1] + uy * r1, x2 = b[0] - ux * r2, y2 = b[1] - uy * r2;
      var cx = (x1 + x2) / 2 - uy * bend, cy = (y1 + y2) / 2 + ux * bend;
      return "M" + x1 + "," + y1 + " Q" + cx + "," + cy + " " + x2 + "," + y2;
    }
    function rad(n) { return info[n] && info[n].kind === "factor" ? 27 : info[n] && info[n].kind === "parameter" ? 34 : 24; }
    edges.forEach(function (e, i) {
      var a = POS[e.from], b = POS[e.to]; if (!a || !b || e.from === e.to) return;
      var key = e.from + ">" + e.to, rev = e.to + ">" + e.from, st = KINDS[e.kind] || KINDS.transition;
      var attrs = { fill: "none", stroke: css(st.color), "stroke-width": st.w, "stroke-dasharray": st.dash, "marker-end": "url(#ar-" + (KINDS[e.kind] ? e.kind : "transition") + ")" };
      var d;
      if (/^mission\.u_option$/.test(e.from) && /^rover\./.test(e.to)) {
        // the option edge between tiers runs down the left margin
        var tx = e.to === "rover.x" ? 124 : 330, ty = e.to === "rover.x" ? 440 : 528, lane = e.to === "rover.x" ? 24 : 36;
        if (e.to === "rover.img") { tx = b[0] - 36; ty = b[1] + 14; }
        d = "M" + (a[0] - 24) + "," + a[1] + " C" + (lane - 10) + "," + a[1] + " " + lane + "," + (a[1] + 20) + " " + lane + "," + (a[1] + 90) + " L" + lane + "," + (e.to === "rover.x" ? ty - 14 : ty + 40) +
          (e.to === "rover.x" ? " C" + lane + "," + ty + " " + (lane + 40) + "," + ty + " " + (b[0] - 28) + "," + ty : " C" + lane + "," + (ty + 62) + " " + (b[0] - 120) + "," + (ty + 62) + " " + (b[0] - 18) + "," + (ty + 20));
        el("path", Object.assign({ d: d }, attrs), svg);
        return;
      }
      if (done[rev] && e.kind === "transition") return;
      var recip = e.kind === "transition" && edges.some(function (o) { return o.from + ">" + o.to === rev && o.kind === "transition"; });
      done[key] = true;
      var bend = BEND[key] != null ? BEND[key] : 0;
      if (recip) { bend = (key === "rover.x>rover.terrain" || key === "rover.terrain>rover.x") ? 78 : 20; attrs["marker-start"] = "url(#as-transition)"; }
      var r1 = rad(e.from), r2 = rad(e.to) + 3;
      if (recip && bend > 40) {
        // x <-> terrain passes over the energy node: an arc above the row
        var xa = POS[key === "rover.x>rover.terrain" ? e.from : e.to], xb = POS[key === "rover.x>rover.terrain" ? e.to : e.from];
        d = "M" + (xa[0] + 8) + "," + (xa[1] - 26) + " C" + (xa[0] + 40) + "," + (xa[1] - 82) + " " + (xb[0] - 40) + "," + (xb[1] - 82) + " " + (xb[0] - 8) + "," + (xb[1] - 28);
        if (key === "rover.terrain>rover.x") return;
      } else d = pathFor(a, b, bend, r1, r2);
      el("path", Object.assign({ d: d }, attrs), svg);
    });
    Object.keys(POS).forEach(function (name) {
      var p = POS[name], i = info[name]; if (!i) return;
      var g = el("g", { "data-node": name }, svg), label = short(name), nd = i.nd, fs = label.length > 9 ? 11 : 12.5;
      var fill = css("--bg"), stroke = css("--fg");
      if (i.kind === "factor") el("circle", { cx: p[0], cy: p[1], r: 26, fill: fill, stroke: stroke, "stroke-width": 1.6 }, g);
      else if (i.kind === "modality") el("rect", { x: p[0] - 32, y: p[1] - 17, width: 64, height: 34, fill: fill, stroke: stroke, "stroke-width": 1.6 }, g);
      else if (i.kind === "parameter") {
        el("rect", { x: p[0] - 52, y: p[1] - 24, width: 104, height: 48, rx: 10, fill: css("--c3"), "fill-opacity": 0.16, stroke: css("--c3"), "stroke-width": 2.6 }, g);
      } else el("rect", { x: p[0] - 38, y: p[1] - 15, width: 76, height: 30, rx: 15, fill: fill, stroke: css("--c2"), "stroke-width": 1.8 }, g);
      el("text", { x: p[0], y: p[1] + (i.kind === "parameter" ? -3 : 4), "text-anchor": "middle", "font-size": fs, "font-weight": i.kind === "parameter" ? 700 : 400 }, g, i.kind === "parameter" ? name : label);
      if (i.kind === "parameter") {
        el("text", { x: p[0], y: p[1] + 11, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, g, "parameter variable");
        el("text", { x: p[0] - 58, y: p[1] - 2, "text-anchor": "end", "font-size": 10.5, class: "muted" }, g, "one Dirichlet table,");
        el("text", { x: p[0] - 58, y: p[1] + 11, "text-anchor": "end", "font-size": 10.5, class: "muted" }, g, "3 kinds × 256 cells");
      } else if (i.kind === "control") {
        el("text", { x: p[0], y: p[1] + 28, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, g, nd.kind === "level" ? nd.levels + " levels" : "continuous");
        el("text", { x: p[0], y: p[1] + 40, "text-anchor": "middle", "font-size": 10.5, class: "now" }, g, "");
      } else {
        var sub = FAM[nd.family] || pretty(nd.family || "");
        if (i.kind === "factor") sub += nd.levels ? " · " + nd.levels : " · dim " + nd.dim;
        el("text", { x: p[0], y: p[1] + (i.kind === "factor" ? 40 : 31), "text-anchor": "middle", "font-size": 10.5, class: "muted" }, g, sub);
      }
    });
    var lg = "", present = {};
    edges.forEach(function (e) { present[e.kind] = true; });
    Object.keys(KINDS).forEach(function (k) {
      if (!present[k]) return;
      var st = KINDS[k];
      lg += "<span><svg class='lsw' viewBox='0 0 28 10'><line x1='1' x2='27' y1='5' y2='5' stroke='" + css(st.color) + "' stroke-width='" + st.w + "' stroke-dasharray='" + st.dash + "'/></svg>" + esc(st.label) + "</span>";
    });
    lg += "<span>circle: state factor</span><span>square: modality</span><span>pill: control</span>";
    $("graphlegend").innerHTML = lg;
    updateGraph();
  }
  function updateGraph() {
    var t = tick();
    document.querySelectorAll("#graph g[data-node]").forEach(function (g) {
      var n = g.getAttribute("data-node"), shape = g.firstElementChild, now = g.querySelector(".now");
      if (n === "mission.u_option") {
        shape.setAttribute("fill", optColor(t.option)); shape.setAttribute("fill-opacity", 0.4);
        if (now) now.textContent = "now: " + optLabel(t.option);
      }
    });
  }

  // ---- the field canvas ----
  function computeWindow() {
    var f = B.meta.field, pad = 0.04 * Math.max(f[0][1] - f[0][0], f[1][1] - f[1][0]);
    return { x0: f[0][0] - pad, x1: f[0][1] + pad, y0: f[1][0] - pad, y1: f[1][1] + pad };
  }
  function setupCanvas(c) {
    var dpr = window.devicePixelRatio || 1, w = c.clientWidth || 520;
    var sc = w / (WIN.x1 - WIN.x0), h = Math.round((WIN.y1 - WIN.y0) * sc);
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    var g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {
      g: g, w: w, h: h, sc: sc, dpr: dpr,
      X: function (x) { return (x - WIN.x0) * sc; },
      Y: function (y) { return h - (y - WIN.y0) * sc; }
    };
  }
  function mapAt(m, ix, iy) { return m[ix][iy]; }
  function contours(g, m, W) {
    // marching squares on the height map, levels every 0.2
    var xs = W.xs, ys = W.ys, H = W.height, nx = xs.length, ny = ys.length;
    var SEG = [[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], [[3, 0], [1, 2]], [[0, 2]], [[3, 2]], [[2, 3]], [[0, 2]], [[0, 1], [2, 3]], [[1, 2]], [[1, 3]], [[0, 1]], [[3, 0]], []];
    g.beginPath();
    for (var lv = 0.2; lv < 1.5; lv += 0.2) {
      for (var i = 0; i < nx - 1; i++) for (var j = 0; j < ny - 1; j++) {
        var c = [H[i][j], H[i + 1][j], H[i + 1][j + 1], H[i][j + 1]], px = [xs[i], xs[i + 1], xs[i + 1], xs[i]], py = [ys[j], ys[j], ys[j + 1], ys[j + 1]];
        var idx = (c[0] > lv ? 1 : 0) | (c[1] > lv ? 2 : 0) | (c[2] > lv ? 4 : 0) | (c[3] > lv ? 8 : 0);
        var segs = SEG[idx]; if (!segs.length) continue;
        var pt = function (e) {
          var a = [0, 1, 2, 3][e], b = (e + 1) % 4, f = (lv - c[a]) / (c[b] - c[a]);
          return [m.X(px[a] + f * (px[b] - px[a])), m.Y(py[a] + f * (py[b] - py[a]))];
        };
        segs.forEach(function (s) { var p = pt(s[0]), q = pt(s[1]); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); });
      }
    }
    g.stroke();
  }
  function background(m) {
    // the true ground, an optional wash, and the height contours, cached per size, layer and theme
    var key = [m.w, m.h, S.layer, css("--tf"), css("--bg")].join("|");
    if (BG && BG.key === key) return BG.canvas;
    var W = B.meta.world, nx = W.xs.length, ny = W.ys.length, oc = document.createElement("canvas");
    oc.width = Math.round(m.w * m.dpr); oc.height = Math.round(m.h * m.dpr);
    var g = oc.getContext("2d"); g.setTransform(m.dpr, 0, 0, m.dpr, 0, 0);
    g.fillStyle = css("--panel"); g.fillRect(0, 0, m.w, m.h);
    var f = B.meta.field, step = W.xs[1] - W.xs[0], stepy = W.ys[1] - W.ys[0];
    var rgb = [0, 1, 2].map(function (c) { return hexrgb(terrBg(c)); });
    var lmin = Infinity, lmax = -Infinity, smin = Infinity, smax = -Infinity;
    for (var i = 0; i < nx; i++) for (var j = 0; j < ny; j++) {
      lmin = Math.min(lmin, W.light[i][j]); lmax = Math.max(lmax, W.light[i][j]);
      smin = Math.min(smin, W.sd_fix[i][j]); smax = Math.max(smax, W.sd_fix[i][j]);
    }
    var wash = hexrgb(css("--washfix").slice(0, 7));
    for (i = 0; i < nx; i++) for (j = 0; j < ny; j++) {
      var x0 = m.X(W.xs[i] - step / 2), x1 = m.X(W.xs[i] + step / 2), y1 = m.Y(W.ys[j] - stepy / 2), y0 = m.Y(W.ys[j] + stepy / 2);
      var c = rgb[W.terrain[i][j]];
      g.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      g.fillRect(x0, y0, x1 - x0 + 0.6, y1 - y0 + 0.6);
      if (S.layer === "light") {
        var n = (W.light[i][j] - lmin) / (lmax - lmin);
        g.fillStyle = "rgba(0,0,0," + (0.55 * (1 - n)).toFixed(3) + ")"; g.fillRect(x0, y0, x1 - x0 + 0.6, y1 - y0 + 0.6);
        g.fillStyle = "rgba(255,236,150," + (0.5 * n).toFixed(3) + ")"; g.fillRect(x0, y0, x1 - x0 + 0.6, y1 - y0 + 0.6);
      } else if (S.layer === "fix") {
        var q = (W.sd_fix[i][j] - smin) / (smax - smin);
        g.fillStyle = "rgba(" + wash.join(",") + "," + (0.7 * q).toFixed(3) + ")"; g.fillRect(x0, y0, x1 - x0 + 0.6, y1 - y0 + 0.6);
      }
    }
    g.strokeStyle = css("--contour"); g.lineWidth = 0.9; contours(g, m, W);
    g.strokeStyle = css("--muted"); g.lineWidth = 1.2; g.strokeRect(m.X(f[0][0]), m.Y(f[1][1]), (f[0][1] - f[0][0]) * m.sc, (f[1][1] - f[1][0]) * m.sc);
    BG = { key: key, canvas: oc };
    return oc;
  }
  function unpackMap(t) {
    if (t.map && t.map._u8) return t.map._u8;
    var s = t.map, out;
    if (typeof s === "string") { var b = atob(s); out = new Uint8Array(b.length); for (var i = 0; i < b.length; i++) out[i] = b.charCodeAt(i); }
    else { out = new Uint8Array(s.length * 3); s.forEach(function (r, i) { out[3 * i] = r[0]; out[3 * i + 1] = r[1]; out[3 * i + 2] = r[2]; }); }
    t._u8 = out; return out;
  }
  function drawMap(m, t) {
    var g = m.g, gr = B.meta.grid, nx = gr.shape[0], ny = gr.shape[1], dx = (gr.high[0] - gr.low[0]) / nx, dy = (gr.high[1] - gr.low[1]) / ny;
    var u = unpackMap(t);
    for (var ix = 0; ix < nx; ix++) for (var iy = 0; iy < ny; iy++) {
      var c = ix * ny + iy;
      if (t.counts && t.counts[c] < 0.02) continue;
      var p = [u[3 * c] / 255, u[3 * c + 1] / 255, u[3 * c + 2] / 255], k = argmax(p), a = clamp((p[k] - 0.34) / 0.66, 0, 1);
      if (a <= 0.02) continue;
      g.globalAlpha = 0.12 + 0.7 * a; g.fillStyle = terrColor(k);
      var x0 = m.X(gr.low[0] + ix * dx), x1 = m.X(gr.low[0] + (ix + 1) * dx), y1 = m.Y(gr.low[1] + iy * dy), y0 = m.Y(gr.low[1] + (iy + 1) * dy);
      g.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
    }
    g.globalAlpha = 1;
    // cell lines, faint
    g.strokeStyle = css("--cellline"); g.lineWidth = 0.5; g.beginPath();
    for (ix = 0; ix <= nx; ix++) { g.moveTo(m.X(gr.low[0] + ix * dx), m.Y(gr.low[1])); g.lineTo(m.X(gr.low[0] + ix * dx), m.Y(gr.high[1])); }
    for (iy = 0; iy <= ny; iy++) { g.moveTo(m.X(gr.low[0]), m.Y(gr.low[1] + iy * dy)); g.lineTo(m.X(gr.high[0]), m.Y(gr.low[1] + iy * dy)); }
    g.stroke();
  }
  function drawSectors(m, t) {
    var g = m.g, f = B.meta.field, sp = B.meta.tiers.zone_split, k = sectorOf(t.option);
    var xr = [[f[0][0], sp[0]], [sp[0], f[0][1]]], yr = [[f[1][0], sp[1]], [sp[1], f[1][1]]];
    if (k >= 0) {
      var r = [xr[k % 2], yr[Math.floor(k / 2)]];
      g.fillStyle = optColor(t.option); g.globalAlpha = 0.2; g.fillRect(m.X(r[0][0]), m.Y(r[1][1]), (r[0][1] - r[0][0]) * m.sc, (r[1][1] - r[1][0]) * m.sc); g.globalAlpha = 1;
      g.strokeStyle = optColor(t.option); g.lineWidth = 3; g.strokeRect(m.X(r[0][0]) + 1.5, m.Y(r[1][1]) + 1.5, (r[0][1] - r[0][0]) * m.sc - 3, (r[1][1] - r[1][0]) * m.sc - 3);
    }
    g.strokeStyle = css("--fg"); g.globalAlpha = 0.7; g.lineWidth = 1.4; g.setLineDash([6, 4]); g.beginPath();
    g.moveTo(m.X(sp[0]), m.Y(f[1][0])); g.lineTo(m.X(sp[0]), m.Y(f[1][1]));
    g.moveTo(m.X(f[0][0]), m.Y(sp[1])); g.lineTo(m.X(f[0][1]), m.Y(sp[1])); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    g.font = "600 12px system-ui, sans-serif"; g.textAlign = "left";
    for (var z = 0; z < 4; z++) {
      var rx = xr[z % 2][0], ry = yr[Math.floor(z / 2)][1];
      var tx = m.X(rx) + 6, ty = m.Y(ry) + 16, txt = "sector " + (z + 1);
      g.lineWidth = 3; g.strokeStyle = css("--panel"); g.strokeText(txt, tx, ty); g.fillStyle = css("--fg"); g.fillText(txt, tx, ty);
    }
  }
  function ellipse(m, comp, color, alpha) {
    var g = m.g, a = comp.cov[0][0], b = comp.cov[0][1], d = comp.cov[1][1];
    var tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    var l1 = tr / 2 + disc, l2 = Math.max(tr / 2 - disc, 1e-10), th = 0.5 * Math.atan2(2 * b, a - d);
    g.beginPath(); g.ellipse(m.X(comp.mu[0]), m.Y(comp.mu[1]), Math.max(4, 2 * Math.sqrt(l1) * m.sc), Math.max(4, 2 * Math.sqrt(l2) * m.sc), -th, 0, 2 * Math.PI);
    g.globalAlpha = alpha * 0.35; g.fillStyle = color; g.fill();
    g.globalAlpha = Math.min(1, alpha * 1.2 + 0.2); g.strokeStyle = color; g.lineWidth = 1.5; g.stroke();
    g.globalAlpha = 1;
  }
  function planPath(t, p) {
    var dy = B.meta.dynamics, gain = dy.gain, keep = dy.keep, dt = dy.dt;
    var terr = argmax(t.belief.terrain), v = (t.belief.velocity || [0, 0]).slice(), x = p.start[0], y = p.start[1], pts = [[x, y]];
    p.v_plan.forEach(function (a) {
      x += dt * v[0]; y += dt * v[1];
      v = [keep[terr] * v[0] + gain[terr] * a[0], keep[terr] * v[1] + gain[terr] * a[1]];
      pts.push([x, y]);
    });
    return pts;
  }
  function drawPlans(m, t) {
    var g = m.g;
    t.policies.forEach(function (p, i) {
      var ch = i === t.chosen, pts = planPath(t, p);
      g.beginPath(); pts.forEach(function (q, k) { k ? g.lineTo(m.X(q[0]), m.Y(q[1])) : g.moveTo(m.X(q[0]), m.Y(q[1])); });
      g.globalAlpha = ch ? 1 : 0.6; g.strokeStyle = ch ? css("--plan") : css("--fg"); g.lineWidth = ch ? 4 : 1.8; g.lineCap = "round"; g.lineJoin = "round"; g.stroke();
      if (ch) { var e = pts[pts.length - 1]; g.beginPath(); g.arc(m.X(e[0]), m.Y(e[1]), 4, 0, 2 * Math.PI); g.fillStyle = css("--plan"); g.fill(); }
      g.globalAlpha = 1;
    });
  }
  function drawRover(m, t) {
    var g = m.g, x = m.X(t.true_x[0]), y = m.Y(t.true_x[1]), vx = t.true_x[2], vy = t.true_x[3], sp = Math.hypot(vx, vy);
    g.save(); g.translate(x, y);
    if (sp > 0.03) { var ang = Math.atan2(-vy, vx); g.rotate(ang); }
    g.beginPath(); g.moveTo(13, 0); g.lineTo(-7, -7); g.lineTo(-3, 0); g.lineTo(-7, 7); g.closePath();
    g.fillStyle = css("--fg"); g.fill(); g.strokeStyle = css("--bg"); g.lineWidth = 2; g.stroke();
    g.restore();
  }
  function drawField() {
    var c = $("field"), m = setupCanvas(c), g = m.g, r = run().ticks, t = r[S.t];
    g.drawImage(background(m), 0, 0, m.w, m.h);
    drawMap(m, t); drawSectors(m, t);
    g.save(); g.setLineDash([2, 5]); g.lineCap = "round"; g.globalAlpha = 0.7; g.strokeStyle = css("--fg"); g.lineWidth = 1.4; g.beginPath();
    for (var i = 0; i <= S.t; i++) { var p = r[i].true_x; i ? g.lineTo(m.X(p[0]), m.Y(p[1])) : g.moveTo(m.X(p[0]), m.Y(p[1])); }
    g.stroke(); g.restore();
    var f = B.meta.field;
    g.save(); g.beginPath(); g.rect(m.X(f[0][0]), m.Y(f[1][1]), (f[0][1] - f[0][0]) * m.sc, (f[1][1] - f[1][0]) * m.sc); g.clip();
    drawPlans(m, t);
    g.restore();
    if (t.fix) {
      var fx = m.X(t.fix[0]), fy = m.Y(t.fix[1]);
      g.strokeStyle = css("--c2"); g.lineWidth = 2; g.beginPath();
      g.moveTo(fx - 4, fy - 4); g.lineTo(fx + 4, fy + 4); g.moveTo(fx - 4, fy + 4); g.lineTo(fx + 4, fy - 4); g.stroke();
    }
    drawRover(m, t);
    t.belief.components.forEach(function (cm) { ellipse(m, cm, css("--ell"), cm.w); });
    drawHud(t);
    drawCamera(t);
  }
  function drawHud(t) {
    var e = t.true_energy, band = e.band, terr = t.belief.terrain, best = argmax(terr);
    var pinned = !twoLevel();
    var opt = optLabel(t.option) + (pinned ? " (pinned)" : "");
    var bar = "<span class='hbar' title='energy level " + e.level.toFixed(2) + " of 4'>";
    for (var b = 0; b < 4; b++) bar += "<i style='width:25%;background:" + (b <= band ? css("--c3") : "transparent") + ";opacity:" + (b < band ? 0.55 : 1) + "'></i>";
    bar += "</span>";
    $("hud").innerHTML =
      "<div><span class='hl'>tick</span><b>" + t.t + "</b></div>" +
      "<div><span class='hl'>option in force</span><b>" + esc(opt) + "</b></div>" +
      "<div><span class='hl'>energy</span>" + bar + " <b>" + esc(nm("energy", band)) + "</b> " + e.level.toFixed(2) + (t.stalled ? " <span class='approx'>stalled</span>" : "") + "</div>" +
      "<div><span class='hl'>terrain under the rover</span>true <b>" + esc(nm("terrain", t.true_terrain)) + "</b>, believed " + esc(nm("terrain", best)) + " " + terr[best].toFixed(2) + "</div>" +
      "<div><span class='hl'>coverage</span><b>" + t.coverage.toFixed(2) + "</b> of cells mapped, <b>" + t.seen.toFixed(2) + "</b> seen</div>";
    var sw = function (c) { return "<i style='background:" + c + "'></i>"; };
    $("fieldlegend").innerHTML =
      "<span>learned map: " + [0, 1, 2].map(function (c) { return sw(terrColor(c)) + esc(nm("terrain", c)); }).join(" ") + "</span>" +
      "<span>true ground: " + [0, 1, 2].map(function (c) { return sw(terrBg(c)) + esc(nm("terrain", c)); }).join(" ") + "</span>" +
      "<span>" + sw(css("--plan")) + "chosen plan (bold), other starts</span><span>" + sw(css("--ell")) + "position belief, 2&sigma; (drawn at least 4 px)</span><span>&times; satellite fix</span><span>dotted: true path</span>" +
      (S.layer === "light" ? "<span>brighter: more sunlight</span>" : S.layer === "fix" ? "<span>purple: noisier fix (terrain blocks the sky)</span>" : "");
  }

  // ---- the camera ----
  function bytes(x, n) {
    if (typeof x !== "string") { var o = new Uint8Array(n * n); for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) o[i * n + j] = x[i][j]; return o; }
    var s = atob(x), out = new Uint8Array(s.length);
    for (var k = 0; k < s.length; k++) out[k] = s.charCodeAt(k);
    return out;
  }
  function paint(c, data, n) {
    var g = c.getContext("2d"), im = g.createImageData(n, n);
    for (var i = 0; i < n * n; i++) { im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = data[i]; im.data[4 * i + 3] = 255; }
    g.putImageData(im, 0, 0);
  }
  function drawCamera(t) {
    var host = $("camgrid"), n = (B.meta.image && B.meta.image.side) || 48;
    if (!host.children.length) {
      [["observed image", "obs"], ["decoder prediction at the belief", "rec"], ["terrain readout", "read"]].forEach(function (p, i) {
        var d = document.createElement("div"); d.className = "cam";
        d.innerHTML = (i < 2 ? "<div class='camc'><canvas id='cam" + i + "' width='" + n + "' height='" + n + "' role='img' aria-label='" + p[0] + "'></canvas>" + (i === 0 ? "<span class='skylab'>sky</span>" : "") + "</div>" : "<div class='readout' id='cam2' role='img' aria-label='" + p[0] + "'></div>") +
          "<span class='cl'>" + p[0] + "</span><span class='cv' id='camv" + i + "'></span>";
        host.appendChild(d);
      });
    }
    paint($("cam0"), bytes(t.img, n), n);
    paint($("cam1"), bytes(t.recon || t.img, n), n);
    var h = "";
    t.belief.terrain.forEach(function (p, i) {
      var isTrue = i === t.true_terrain;
      h += "<div class='rb" + (isTrue ? " true" : "") + "'><span class='rn'>" + esc(nm("terrain", i)) + (isTrue ? " (true)" : "") + "</span><span class='rp'>" + p.toFixed(2) + "</span><span class='bar'><i style='width:" + (100 * p).toFixed(1) + "%;background:" + terrColor(i) + "'></i></span></div>";
    });
    $("cam2").innerHTML = h;
    var cam = t.camera || {};
    $("camv1").textContent = cam.log_evidence != null ? "log evidence " + cam.log_evidence.toFixed(0) : "";
    $("camv0").textContent = "looking along +y";
  }

  // ---- timeline ----
  function drawTimeline() {
    var host = $("timeline"), T = run().ticks, n = T.length, C = cache(), O = B.runs[other()].ticks;
    var W = Math.max(300, host.clientWidth - 16), L = W < 480 ? 46 : 64, R = 40, sx = function (i) { return L + i / n * (W - L - R); };
    host.innerHTML = "";
    var yS = 14, hS = 26, yE = 82, hE = 76, yC = 194, hC = 100, H = yC + hC + 28;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": "Timeline of the option in force, the energy level and coverage" }, host);
    svg.style.height = "auto";
    function lab(txt, y, anchor) { el("text", { x: 4, y: y, "font-size": 10.5, class: "muted" }, svg, txt); }
    lab("option", yS + 17);
    C.segs.forEach(function (s) {
      var x0 = sx(s.t0), x1 = sx(s.t1);
      el("rect", { x: x0, y: yS, width: x1 - x0, height: hS, fill: optColor(s.option), "fill-opacity": 0.75, stroke: css("--panel"), "stroke-width": 1 }, svg);
      var txt = (!twoLevel() && s.option === B.meta.tiers.pinned_option) ? "pinned: " + optLabel(s.option) : (x1 - x0 > 70 ? optLabel(s.option) : x1 - x0 > 16 ? optShort(s.option) : "");
      if (txt) el("text", { x: (x0 + x1) / 2, y: yS + 17, "text-anchor": "middle", "font-size": 10.5, fill: css("--fg") }, svg, txt);
    });
    if (twoLevel()) T.forEach(function (t, i) { if (t.slow && t.slow.decision) el("line", { x1: sx(i), x2: sx(i), y1: yS - 8, y2: yS, stroke: css("--fg"), "stroke-width": 1.2 }, svg); });
    el("text", { x: L, y: yE - 9, "font-size": 10.5, class: "muted" }, svg, "energy level (true), against the four bands");
    el("text", { x: L, y: yC - 6, "font-size": 10.5, class: "muted" }, svg, "coverage: fraction of cells mapped");
    // energy
    var ey = function (v) { return yE + hE - (v / 4) * hE; };
    for (var b = 0; b <= 4; b++) el("line", { x1: L, x2: W - R, y1: ey(b), y2: ey(b), stroke: css("--line"), "stroke-width": 0.8 }, svg);
    for (b = 0; b < 4; b++) el("text", { x: L - 4, y: ey(b + 0.5) + 3, "text-anchor": "end", "font-size": 9.5, class: "muted" }, svg, nm("energy", b));
    var d = "";
    T.forEach(function (t, i) { d += (i ? "L" : "M") + sx(i + 0.5).toFixed(1) + "," + ey(t.true_energy.level).toFixed(1); });
    el("path", { d: d, fill: "none", stroke: css("--c3"), "stroke-width": 2 }, svg);
    // coverage
    var cmax = 0.5, cy = function (v) { return yC + hC - clamp(v / cmax, 0, 1) * hC; };
    [0, 0.25, 0.5].forEach(function (v) {
      el("line", { x1: L, x2: W - R, y1: cy(v), y2: cy(v), stroke: css("--line"), "stroke-width": 0.8 }, svg);
      el("text", { x: L - 4, y: cy(v) + 3, "text-anchor": "end", "font-size": 9.5, class: "muted" }, svg, v.toFixed(2));
    });
    function curve(ts, color, dash, w) {
      var s = ""; ts.forEach(function (t, i) { s += (i ? "L" : "M") + sx(i + 0.5).toFixed(1) + "," + cy(t.coverage).toFixed(1); });
      el("path", { d: s, fill: "none", stroke: color, "stroke-width": w, "stroke-dasharray": dash }, svg);
      var e = ts[ts.length - 1]; el("text", { x: W - R + 4, y: cy(e.coverage) + 3.5, "font-size": 10.5, fill: color, "font-weight": 600 }, svg, e.coverage.toFixed(2));
    }
    curve(O, css("--muted"), "5 4", 2);
    curve(T, css("--accent"), "", 2.4);
    for (var k = 0; k <= n; k += 100) {
      el("line", { x1: sx(k), x2: sx(k), y1: yC + hC, y2: yC + hC + 4, stroke: css("--muted") }, svg);
      el("text", { x: sx(k), y: yC + hC + 16, "text-anchor": "middle", "font-size": 10, class: "muted" }, svg, k === n ? "tick " + n : k);
    }
    // cursor
    var cx = sx(S.t + 0.5);
    el("line", { x1: cx, x2: cx, y1: yS - 8, y2: yC + hC, stroke: css("--fg"), "stroke-width": 1.4 }, svg);
    el("circle", { cx: cx, cy: ey(T[S.t].true_energy.level), r: 3.5, fill: css("--c3"), stroke: css("--bg"), "stroke-width": 1.5 }, svg);
    el("circle", { cx: cx, cy: cy(T[S.t].coverage), r: 3.5, fill: css("--accent"), stroke: css("--bg"), "stroke-width": 1.5 }, svg);
    function pick(ev) {
      var r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) * (W / r.width);
      S.t = clamp(Math.round((x - L) / (W - L - R) * n - 0.5), 0, n - 1); render();
    }
    svg.addEventListener("pointerdown", function (e) { svg.setPointerCapture(e.pointerId); pick(e); svg._d = true; });
    svg.addEventListener("pointermove", function (e) { if (svg._d) pick(e); });
    svg.addEventListener("pointerup", function () { svg._d = false; });
    svg.addEventListener("pointercancel", function () { svg._d = false; });
    var rs = settingRec(), ro = settingRec(other()), tl = "";
    for (var o = 0; o < 5; o++) if (C.segs.some(function (s) { return s.option === o; }) || twoLevel()) tl += "<span><i style='background:" + optColor(o) + "'></i>" + esc(optLabel(o)) + "</span>";
    tl += "<span><svg class='lsw' viewBox='0 0 28 10'><line x1='1' x2='27' y1='5' y2='5' stroke='" + css("--accent") + "' stroke-width='2.4'/></svg>coverage, " + esc(rs.label.toLowerCase()) + "</span>" +
      "<span><svg class='lsw' viewBox='0 0 28 10'><line x1='1' x2='27' y1='5' y2='5' stroke='" + css("--muted") + "' stroke-width='2' stroke-dasharray='5 4'/></svg>coverage, " + esc(ro.label.toLowerCase()) + "</span>" +
      "<span><svg class='lsw' viewBox='0 0 28 10'><line x1='1' x2='27' y1='5' y2='5' stroke='" + css("--c3") + "' stroke-width='2'/></svg>energy level</span>";
    $("tlegend").innerHTML = tl;
  }

  // ---- the scheduler's decision: G by first option, split into novelty and energy risk ----
  var DECR = null;
  function decRange() {
    if (DECR) return DECR;
    var lo = 0, hi = 0;
    B.runs[Object.keys(B.runs).filter(function (k) { return settingRec(k).tiers === 2; })[0]].ticks.forEach(function (t) {
      if (!t.slow || !t.slow.decision) return;
      t.slow.efe_by_option.forEach(function (g) { lo = Math.min(lo, g); hi = Math.max(hi, g); });
      t.slow.policies.forEach(function (p) { var s = leafSums(p); lo = Math.min(lo, s.novelty); hi = Math.max(hi, s.risk); });
    });
    return (DECR = [lo, hi]);
  }
  function leafSums(p) {
    var nov = 0, risk = 0;
    Object.keys(p.terms).forEach(function (tm) {
      Object.keys(p.terms[tm]).forEach(function (lf) {
        var v = (p.terms[tm][lf][0] + p.terms[tm][lf][1]) / 2;
        if (tm === "infogain") nov += v; else risk += v;
      });
    });
    return { novelty: nov, risk: risk };
  }
  function drawDecision() {
    var host = $("decision"), note = $("decnote"), cap = $("deccap"), lg = $("declegend");
    host.innerHTML = ""; lg.innerHTML = "";
    if (!twoLevel()) {
      note.textContent = "In this setting the option is pinned to " + optLabel(B.meta.tiers.pinned_option) + ", so the scheduler does not run and there is no decision to show.";
      cap.textContent = ""; host.hidden = true; return;
    }
    host.hidden = false;
    var d = decTick();
    var sgn = "Sign convention: every term enters G with its sign applied, so novelty appears negative (lower G is better) and energy risk positive.";
    var chosen = d.slow.option;
    note.textContent = "Decision at tick " + d.t + ": " + optLabel(chosen) + ". The scheduler scores every sequence of two options; each bar below is the best sequence that starts with that option, split into its two objective terms (novelty of the map, which is the information the plan expects to gain about the table, and energy risk), with a mark at their sum, G.";
    var rg = decRange(), lo = rg[0], hi = rg[1], pad = (hi - lo) * 0.05; lo -= pad; hi += pad;
    var Wd = Math.max(300, host.clientWidth - 16), Lx = Wd < 520 ? 96 : 168, Rx = 70, rowH = 34, H = 10 + 5 * rowH + 30;
    var sx = function (v) { return Lx + (v - lo) / (hi - lo) * (Wd - Lx - Rx); };
    var svg = el("svg", { viewBox: "0 0 " + Wd + " " + H, width: Wd, height: H, role: "img", "aria-label": "Expected free energy by first option, split into novelty and energy risk" }, host);
    svg.style.height = "auto";
    var step = 10;
    for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      el("line", { x1: sx(v), x2: sx(v), y1: 6, y2: H - 24, stroke: css("--line"), "stroke-width": v === 0 ? 1.6 : 0.8 }, svg);
      el("text", { x: sx(v), y: H - 8, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, svg, v);
    }
    var best = {};
    d.slow.policies.forEach(function (p) {
      var k = p.options[0], m = (p.G[0] + p.G[1]) / 2;
      if (!best[k] || m < best[k].m) best[k] = { p: p, m: m };
    });
    var missing = false;
    for (var k = 0; k < 5; k++) {
      var y = 10 + k * rowH, ch = k === chosen, G = d.slow.efe_by_option[k], b = best[k];
      if (ch) el("rect", { x: 1, y: y, width: Wd - 2, height: rowH - 2, rx: 3, fill: "none", stroke: css("--accent"), "stroke-width": 1.5 }, svg);
      var name = optLabel(k) + (b ? "" : "");
      el("text", { x: 8, y: y + rowH / 2 + 0, "font-size": 11.5, "font-weight": ch ? 700 : 400 }, svg, (ch ? "▸ " : "") + name);
      if (b) {
        var s = leafSums(b.p), then = b.p.options.slice(1).map(optLabel).join(", ");
        el("text", { x: 8, y: y + rowH / 2 + 12, "font-size": 9.5, class: "muted" }, svg, "then " + then);
        el("rect", { x: Math.min(sx(s.novelty), sx(0)), y: y + 6, width: Math.abs(sx(0) - sx(s.novelty)), height: rowH - 16, fill: termColor("infogain"), "fill-opacity": 0.8 }, svg);
        el("rect", { x: Math.min(sx(s.risk), sx(0)), y: y + 6, width: Math.max(1, Math.abs(sx(s.risk) - sx(0))), height: rowH - 16, fill: termColor("risk"), "fill-opacity": 0.85 }, svg);
      } else {
        missing = true;
        el("text", { x: 8, y: y + rowH / 2 + 12, "font-size": 9.5, class: "muted" }, svg, "terms not kept");
        el("rect", { x: Math.min(sx(G), sx(0)), y: y + 6, width: Math.abs(sx(0) - sx(G)), height: rowH - 16, fill: css("--muted"), "fill-opacity": 0.3, stroke: css("--muted"), "stroke-dasharray": "3 2" }, svg);
      }
      el("path", { d: "M" + sx(G) + "," + (y + 3) + " l5,11 l-5,11 l-5,-11 z", fill: css("--fg"), stroke: css("--bg"), "stroke-width": 1 }, svg);
      el("text", { x: Wd - Rx + 8, y: y + rowH / 2 + 4, "font-size": 11, "font-family": "ui-monospace,Menlo,monospace" }, svg, "G " + fmt(G, 1).replace("+", ""));
    }
    lg.innerHTML = "<span><i style='background:" + termColor("infogain") + "'></i>novelty of the map</span><span><i style='background:" + termColor("risk") + "'></i>energy risk</span><span>&#9670; G, the sum</span><span>outlined: the option chosen</span>";
    cap.textContent = sgn + " Sequences are scored over the two options of the plan; the chart's axis is fixed over the run so decisions can be compared." + " Each bar is the best sequence that starts with that option.";
  }

  // ---- objectives as intervals ----
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
  function approx(tm, lf) { var a = S.tier === "rover" && tick().approximate; return !!(a && a[tm] && a[tm][lf]); }
  function anyApprox() { var a = S.tier === "rover" && tick().approximate; if (!a) return false; for (var tm in a) for (var lf in a[tm]) if (a[tm][lf]) return true; return false; }
  function showTip(ev, it) {
    var h = "<b>" + esc(it.name) + "</b> &nbsp; posterior weight qπ=" + it.q.toFixed(3) + "<br>G " + (Math.abs(it.G[1] - it.G[0]) < 1e-9 ? "= " + it.G[0].toFixed(2) + " (point)" : "[" + it.G[0].toFixed(2) + ", " + it.G[1].toFixed(2) + "]") + "<table>", ap = false;
    leafRows(it.p).forEach(function (r) {
      var a = approx(r[0], r[1]); ap = ap || a;
      h += "<tr><td style='color:" + termColor(r[0]) + "'>" + r[0] + "</td><td>" + esc(r[1]) + (a ? " <span class='ap'>≈</span>" : "") + "</td><td>" + (Math.abs(r[2] - r[3]) < 1e-9 ? r[2].toFixed(2) : "[" + r[2].toFixed(2) + ", " + r[3].toFixed(2) + "]") + "</td></tr>";
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
        items: t.policies.map(function (p, i) { return { name: "start " + p.id, sub: "q=" + p.q_pi.toFixed(2), G: p.G, q: p.q_pi, p: p, chosen: i === t.chosen }; })
      };
    }
    var d = decTick(); if (!d) return { terms: [], items: [], decided: null };
    var items = d.slow.policies.map(function (p, i) {
      return { name: p.options.map(function (o) { return sectorOf(o) >= 0 ? "survey " + (sectorOf(o) + 1) : optLabel(o); }).join(" → "), sub: "q=" + p.q_pi.toFixed(2), G: p.G, q: p.q_pi, p: p, chosen: i === d.slow.chosen };
    });
    items.sort(function (a, b) { return (a.G[0] + a.G[1]) - (b.G[0] + b.G[1]); });
    items.forEach(function (it, i) { it.rank = i + 1; });
    return { terms: Object.keys(d.slow.policies[0].terms), decided: d.t, items: items };
  }
  function objRange(items, terms) {
    var lo = Infinity, hi = -Infinity;
    items.forEach(function (it) {
      lo = Math.min(lo, it.G[0]); hi = Math.max(hi, it.G[1]);
      terms.forEach(function (tm) { var s = termSum(it.p, tm); lo = Math.min(lo, s[0]); hi = Math.max(hi, s[1]); });
    });
    return [lo, hi];
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
    var two = twoLevel();
    $("objnote").textContent = mission
      ? (two ? "Mission tier: the scheduler's option sequences are enumerated exactly, so each G is a point (lower equals upper) and is drawn as a tick, not a bar. Its objective terms are the novelty of the map and the energy risk."
        : "Mission tier: in this setting the option is pinned and the scheduler does not run, so there are no sequences to show. Switch the setting to the scheduler over the rover.")
      : "Rover tier: the gradient planner descends from four warm starts (initial velocity plans) and keeps the best; the bars are the four starts' plans, the acted one outlined, and the intervals the closed-form mixture bounds put around each term." + (anyApprox() ? " Terms that read the camera are computed on the decoder's linearisation at the belief and are marked ≈ in the tooltip." : "");
    $("objcap").textContent = (mission
      ? "Expected free energy of the kept option sequences (the " + items.length + " the page keeps: the best overall and the best for each first option) at the most recent decision (tick " + o.decided + "), best first; lower is better. "
      : "Expected free energy G of the chosen plan at the current tick (lower is better). ") +
      "Hover or focus a " + (mission ? "sequence" : "plan") + " for its per-term, per-leaf values. The chosen " + (mission ? "sequence" : "plan") + " is highlighted. " + (B.meta.sign_convention ? "Sign convention: " + B.meta.sign_convention + "." : "") +
      (mode === "terms" ? " Each G is split into its objective terms, stacked from zero: negative terms to the left, positive to the right; a bracket marks each term's interval." : "");
    var btn = $("objall"); btn.hidden = !(mission && items.length > TOP_N);
    btn.textContent = S.showAll ? "Show top " + TOP_N : "Show all " + items.length + " kept";
    if (mission && !S.showAll && items.length > TOP_N) {
      var top = items.slice(0, TOP_N), ch = items.filter(function (i) { return i.chosen; })[0];
      if (ch && top.indexOf(ch) < 0) top.push(ch);
      items = top;
    }
    host.innerHTML = "";
    if (!items.length) { host.textContent = mission && !two ? "No scheduler in this setting." : "No decision yet."; return; }
    var rg = mode === "terms" ? stackRange(items, terms) : objRange(items, terms), lo = Math.min(rg[0], 0), hi = Math.max(rg[1], 0), padx = (hi - lo) * 0.03 || 1; lo -= padx; hi += padx;
    var W = Math.max(300, host.clientWidth - 16), L = mission ? (W < 520 ? 124 : 190) : 74, R = 14, rowH = mode === "total" ? 36 : 58, top0 = 8, H = top0 + items.length * rowH + 34;
    var sx = function (v) { return L + (v - lo) / (hi - lo) * (W - L - R); };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": "Expected free energy per " + (mission ? "option sequence" : "plan") }, host);
    svg.style.height = "auto";
    var step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 5))); step *= (hi - lo) / step / 5 > 3 ? 5 : (hi - lo) / step / 5 > 1.5 ? 2 : 1;
    for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      el("line", { x1: sx(v), x2: sx(v), y1: top0, y2: H - 26, stroke: css("--line"), "stroke-width": Math.abs(v) < 1e-9 ? 1.6 : 0.8 }, svg);
      el("text", { x: sx(v), y: H - 10, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg, (Math.round(v * 100) / 100) + "");
    }
    function mark(g, iv, y, h, color, op, strong) {
      var w = sx(iv[1]) - sx(iv[0]);
      if (w < 1.5) {
        el("line", { x1: sx(iv[0]), x2: sx(iv[0]), y1: y, y2: y + h, stroke: color, "stroke-opacity": op, "stroke-width": strong ? 4 : 3, "stroke-linecap": "round" }, g);
      } else {
        el("rect", { x: sx(iv[0]), y: y, width: w, height: h, rx: 2, fill: color, "fill-opacity": op }, g);
        var m = (iv[0] + iv[1]) / 2; el("line", { x1: sx(m), x2: sx(m), y1: y - 2, y2: y + h + 2, stroke: css("--fg"), "stroke-width": 2 }, g);
      }
    }
    items.forEach(function (it, i) {
      var y = top0 + i * rowH, ch = it.chosen, p = it.p;
      var g = el("g", { class: "row", tabindex: 0, "aria-label": it.name + ", G " + it.G[0].toFixed(2) + (Math.abs(it.G[0] - it.G[1]) < 1e-9 ? "" : " to " + it.G[1].toFixed(2)) + (ch ? ", chosen" : "") }, svg);
      el("rect", { class: "rowbg", x: 0, y: y, width: W, height: rowH, fill: "transparent" }, g);
      if (ch) el("rect", { x: 1, y: y + 1, width: W - 2, height: rowH - 2, fill: "none", stroke: css("--accent"), "stroke-width": 1.5, rx: 3 }, g);
      var label = (ch ? "▸ " : "") + (mission ? "#" + it.rank + " " : "") + it.name;
      var maxc = Math.floor((L - 12) / 6.2);
      el("text", { x: 8, y: y + rowH / 2 - 1, "font-size": 11.5, "font-weight": ch ? 700 : 400 }, g, label.length > maxc ? label.slice(0, maxc - 1) + "…" : label);
      el("text", { x: 8, y: y + rowH / 2 + 14, "font-size": 9.5, class: "muted" }, g, it.sub);
      if (mode === "total") {
        mark(g, it.G, y + 8, rowH - 16, css(ch ? "--accent" : "--muted"), ch ? 0.9 : 0.55, ch);
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

  // ---- description length ----
  function nats(v) { return v < 1000 ? v.toFixed(2) : Math.round(v).toLocaleString("en-US"); }
  function drawMdl() {
    var tb = document.querySelector("#mdltable tbody");
    tb.innerHTML = "";
    var C = B.mdl.components, mx = 0, tot = 0, exact = 0;
    C.forEach(function (c) { if (c.nats != null) { mx = Math.max(mx, Math.log10(c.nats + 1)); tot += c.nats; if (c.exact) exact += c.nats; } });
    function tierOf(c) { var m = /(?:^|:)(mission|rover)\./.exec(c.name); return m ? m[1] : null; }
    function comp(c, note) {
      var tr = document.createElement("tr"), un = c.nats == null;
      var a = document.createElement("td"); a.textContent = c.name.replace(/^[a-z_]+:/, "").replace(/^(mission|rover)\./, ""); a.title = c.name; tr.appendChild(a);
      var k = document.createElement("td"); k.className = "kind"; k.textContent = pretty(c.kind); tr.appendChild(k);
      var pc = document.createElement("td"); pc.className = "num muted"; pc.textContent = c.entries != null ? c.entries.toLocaleString() : ""; tr.appendChild(pc);
      var n = document.createElement("td"); n.className = "num" + (un ? " unscored" : ""); n.textContent = un ? "unscored" : nats(c.nats); tr.appendChild(n);
      var bc = document.createElement("td"); bc.className = "mdlbarcell";
      if (!un) { var bar = document.createElement("span"); bar.className = "mdlbar"; bar.style.width = (100 * Math.log10(c.nats + 1) / (mx || 1)).toFixed(1) + "%"; bar.title = "log10(nats + 1) = " + Math.log10(c.nats + 1).toFixed(2); bc.appendChild(bar); }
      tr.appendChild(bc);
      var e = document.createElement("td"); e.textContent = (un ? "unscored" : (c.exact ? "exact" : "approximate")) + (note ? "; " + note : ""); if (!un && !c.exact) e.className = "approx"; tr.appendChild(e);
      tb.appendChild(tr);
    }
    function group(title, cs) {
      var h = document.createElement("tr"); h.className = "grp"; h.innerHTML = "<td colspan='6'>" + esc(title) + "</td>"; tb.appendChild(h);
      var sub = 0, ex = 0;
      cs.forEach(function (c) {
        var ref = c.exact && c.entries === 0 && /^transition:/.test(c.name);
        comp(c, ref ? "structure only, the table is " + (B.graph.parameters[0] || {}).name : "");
        if (c.nats != null) { sub += c.nats; if (c.exact) ex += c.nats; }
      });
      var s = document.createElement("tr"); s.className = "grp light";
      s.innerHTML = "<td colspan='3'>" + (/tier/i.test(title) ? "Tier subtotal" : "Subtotal") + "</td><td class='num'>" + nats(sub) + "</td><td></td><td>" + nats(ex) + " of it exact</td>";
      tb.appendChild(s);
    }
    var isParam = function (c) { return c.kind === "parameter"; };
    group("Mission tier (the scheduler)", C.filter(function (c) { return tierOf(c) === "mission" && !isParam(c); }));
    group("Rover tier", C.filter(function (c) { return tierOf(c) === "rover" && !isParam(c); }));
    var h = document.createElement("tr"); h.className = "grp"; h.innerHTML = "<td colspan='6'>The composite</td>"; tb.appendChild(h);
    C.filter(function (c) { return c.kind === "graph"; }).forEach(function (c) { comp(c, "both tiers"); });
    C.filter(isParam).forEach(function (c) { comp(c, "priced once; its two readers pay structure only"); });
    var tr = document.createElement("tr"); tr.className = "grp total";
    tr.innerHTML = "<td colspan='3'>total over scored components</td><td class='num'>" + nats(tot) + "</td><td></td><td>" + nats(exact) + " of it exact</td>";
    tb.appendChild(tr);
    var big = C.slice().sort(function (a, b) { return b.nats - a.nats; })[0];
    $("mdlfoot").textContent = "Most of the total is one component: the camera's likelihood network (" + big.entries.toLocaleString() + " weights, priced by its family alone and so flagged approximate). The exact entries are the graph, the controls, the two structure-only readers and the parameter variable, which is " + (100 * exact / tot).toFixed(1) + "% of the total. Bars are proportional to log10(nats + 1).";
  }

  // ---- state / wiring ----
  function render() {
    $("tick").value = S.t; $("tickout").textContent = S.t;
    drawField(); updateGraph(); drawTimeline(); drawDecision(); drawObjective();
  }
  function setPlaying(on) {
    S.playing = on; $("play").textContent = on ? "Pause" : "Play"; $("play").setAttribute("aria-label", on ? "Pause" : "Play");
    clearInterval(S.timer);
    if (on) S.timer = setInterval(function () { S.t = (S.t + 1) % run().ticks.length; render(); }, 120);
  }
  function fullRedraw() { BG = null; drawGraph(); render(); }
  function hashParams() {
    var o = {}; location.hash.replace(/^#/, "").split("&").forEach(function (kv) { var p = kv.split("="); if (p[0]) o[p[0]] = decodeURIComponent(p[1] || ""); });
    return o;
  }

  function init(b) {
    B = b; tipEl = $("tip"); WIN = computeWindow();
    $("status").hidden = true;
    $("dynrule").textContent = B.meta.dynamics.rule;
    var hp = hashParams();
    if (hp.theme) document.documentElement.setAttribute("data-theme", hp.theme);
    if (B.meta.library_version) document.querySelectorAll("#version, .version").forEach(function (vs) { vs.textContent = "v" + B.meta.library_version; });
    var sel = $("setting");
    B.settings.forEach(function (s) { var o = document.createElement("option"); o.value = s.id; o.textContent = s.label + (s.tiers === 2 ? " (two levels)" : " (one level)"); sel.appendChild(o); });
    S.setting = B.runs[hp.setting] ? hp.setting : B.settings[0].id; sel.value = S.setting;
    var n = run().ticks.length; $("tick").max = n - 1;
    S.t = Math.max(0, Math.min(n - 1, +hp.t || 0));
    if (hp.tier === "rover") { S.tier = "rover"; $("objtier").value = "rover"; } else $("objtier").value = "mission";
    if (hp.view === "terms") $("objmode").value = "terms";
    if (hp.layer && /^(terrain|light|fix)$/.test(hp.layer)) { S.layer = hp.layer; document.querySelector("input[name=layer][value=" + hp.layer + "]").checked = true; }
    sel.addEventListener("change", function () { S.setting = sel.value; S.t = $("restart").checked ? 0 : Math.min(S.t, run().ticks.length - 1); $("tick").max = run().ticks.length - 1; fullRedraw(); });
    $("tick").addEventListener("input", function (e) { S.t = +e.target.value; render(); });
    $("play").addEventListener("click", function () { setPlaying(!S.playing); });
    $("objmode").addEventListener("change", drawObjective);
    $("objtier").addEventListener("change", function (e) { S.tier = e.target.value; drawObjective(); });
    $("objall").addEventListener("click", function () { S.showAll = !S.showAll; drawObjective(); });
    document.querySelectorAll("input[name=layer]").forEach(function (r) { r.addEventListener("change", function () { S.layer = r.value; BG = null; drawField(); }); });
    drawMdl(); fullRedraw();
    var rt; window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(function () { BG = null; render(); }, 100); });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { drawMdl(); fullRedraw(); });
    // nav.js shows one pane at a time; a hidden pane has no size, so redraw when one is shown
    document.addEventListener("hybrid-aif:pane", function () { BG = null; render(); });
    window.__app = { set: function (o) {
      if (o.setting && B.runs[o.setting]) { S.setting = o.setting; $("setting").value = o.setting; $("tick").max = run().ticks.length - 1; BG = null; drawGraph(); }
      if (o.tier) { S.tier = o.tier; $("objtier").value = o.tier; }
      if (o.view) $("objmode").value = o.view;
      if (o.all != null) S.showAll = o.all;
      if (o.layer) { S.layer = o.layer; BG = null; document.querySelector("input[name=layer][value=" + o.layer + "]").checked = true; }
      if (o.t != null) S.t = o.t; render();
    }, state: function () { return { t: S.t, setting: S.setting, playing: S.playing }; } };
    window.__ready = true;
    document.dispatchEvent(new Event("hybrid-aif:ready"));
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
