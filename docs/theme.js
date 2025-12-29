// Theme Toggle Script
(function () {
  const key = "theme";
  const root = document.documentElement;

  function apply(theme) {
    root.setAttribute("data-theme", theme);
    localStorage.setItem(key, theme);
  }

  // Apply saved theme or system preference
  const saved = localStorage.getItem(key);
  if (saved === "light" || saved === "dark") {
    root.setAttribute("data-theme", saved);
  } else {
    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.setAttribute("data-theme", prefersDark ? "dark" : "light");
  }

  // Toggle function
  window.toggleTheme = function () {
    const current = root.getAttribute("data-theme");
    apply(current === "dark" ? "light" : "dark");
  };
})();

// KaTeX Auto-render
document.addEventListener("DOMContentLoaded", function () {
  if (typeof renderMathInElement !== 'undefined') {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false
    });
  }
});
