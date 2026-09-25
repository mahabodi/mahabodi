// MahaBodi site: shared theme toggle (light by default, remembered), mobile menu, code tabs.
(function () {
  var root = document.documentElement;
  try { var t = localStorage.getItem('mb-theme'); if (t) root.setAttribute('data-theme', t); } catch (e) {}
  window.toggleTheme = function () {
    var n = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', n);
    try { localStorage.setItem('mb-theme', n); } catch (e) {}
  };
  window.toggleMenu = function () { document.querySelector('.nav').classList.toggle('open'); };
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.tabs').forEach(function (tabs) {
      tabs.querySelectorAll('button').forEach(function (b) {
        b.addEventListener('click', function () {
          tabs.querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-selected', 'false'); });
          b.setAttribute('aria-selected', 'true');
          var group = tabs.getAttribute('data-group');
          document.querySelectorAll('.panel[data-group="' + group + '"]').forEach(function (p) {
            p.classList.toggle('on', p.id === b.getAttribute('data-p'));
          });
        });
      });
    });
  });
})();
