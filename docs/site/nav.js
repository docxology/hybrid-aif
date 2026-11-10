(function () {
  "use strict";
  // Shows one pane at a time, the one the URL hash names. The hash is "#id", where id is a pane's
  // id or the id of an element inside a pane (a subheading), optionally followed by the viewer's
  // "&key=value" settings; a hash of settings only opens the viewer's pane, and an empty or
  // unknown hash opens the Intro pane. Showing a pane fires "hybrid-aif:pane" on the document so
  // the viewer can redraw at the pane's size.
  var root = document.documentElement;
  var main = document.getElementById("main");
  var sidebar = document.getElementById("sidebar");
  var menu = document.getElementById("menu");
  if (!main || !sidebar) return;
  var panes = Array.prototype.slice.call(main.querySelectorAll(".pane"));
  var links = Array.prototype.slice.call(sidebar.querySelectorAll(".toc a"));
  var narrow = window.matchMedia("(max-width: 900px)");
  var current = null, target = null, lastScroll = null;

  function resolve() {
    var tokens = location.hash.replace(/^#/, "").split("&"), id = null, settings = false;
    tokens.forEach(function (t) {
      if (t.indexOf("=") >= 0) settings = true;
      else if (t && id === null) id = decodeURIComponent(t);
    });
    var el = id ? document.getElementById(id) : null;
    var pane = el && main.contains(el) ? (el.classList.contains("pane") ? el : el.closest(".pane")) : null;
    if (!pane && settings) {
      var field = document.getElementById("field");
      pane = field ? field.closest(".pane") : null;
      el = pane;
    }
    if (!pane) { pane = document.getElementById("intro") || panes[0]; el = pane; }
    return { pane: pane, el: el };
  }

  function scrollToTarget() {
    if (target && target !== current) target.scrollIntoView({ block: "start", behavior: "instant" });
    else window.scrollTo({ top: 0, left: 0, behavior: "instant" });
    lastScroll = window.scrollY;
  }

  function show() {
    var r = resolve();
    target = r.el;
    if (r.pane !== current) {
      panes.forEach(function (p) { p.classList.toggle("is-current", p === r.pane); });
      current = r.pane;
      document.dispatchEvent(new CustomEvent("hybrid-aif:pane", { detail: { id: current.id } }));
    }
    links.forEach(function (a) {
      var id = a.getAttribute("href").slice(1);
      if (a.hasAttribute("data-pane")) {
        if (id === current.id) a.setAttribute("aria-current", "page");
        else a.removeAttribute("aria-current");
      } else {
        a.classList.toggle("is-current", target !== current && id === target.id);
      }
    });
    scrollToTarget();
  }

  function setMenu(open) {
    if (!menu) return;
    sidebar.classList.toggle("open", open);
    menu.setAttribute("aria-expanded", open ? "true" : "false");
    menu.textContent = open ? "Close" : "Menu";
  }

  if (menu) {
    menu.hidden = false;
    menu.addEventListener("click", function () { setMenu(!sidebar.classList.contains("open")); });
  }
  sidebar.addEventListener("click", function (e) {
    var a = e.target.closest ? e.target.closest(".toc a, .brand") : null;
    if (!a || !narrow.matches) return;
    setMenu(false);
    current && current.focus({ preventScroll: true });
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && sidebar.classList.contains("open")) { setMenu(false); menu.focus(); }
  });
  narrow.addEventListener("change", function () { setMenu(false); });
  panes.forEach(function (p) { p.setAttribute("tabindex", "-1"); });
  window.addEventListener("hashchange", show);
  // the browser scrolls to the hash's element itself on load; settle on the pane's own position
  window.addEventListener("load", function () { if (window.scrollY !== lastScroll) scrollToTarget(); });
  // the viewer's first draw can change the height above a subheading: keep it in view, unless
  // the reader has scrolled since
  document.addEventListener("hybrid-aif:ready", function () {
    if (target && target !== current && window.scrollY === lastScroll) scrollToTarget();
  });
  show();
})();
