(() => {
  "use strict";

  const slides = Array.from(document.querySelectorAll(".slide"));
  const stage = document.querySelector(".stage");
  const progress = document.querySelector(".progress-bar");
  const counter = document.querySelector(".counter");
  const prevButton = document.querySelector("[data-action='prev']");
  const nextButton = document.querySelector("[data-action='next']");
  const overviewButton = document.querySelector("[data-action='overview']");
  const fullscreenButton = document.querySelector("[data-action='fullscreen']");
  const notesButton = document.querySelector("[data-action='notes']");
  const printButton = document.querySelector("[data-action='print']");
  const overview = document.querySelector(".overview");
  const overviewGrid = document.querySelector(".overview-grid");
  const overviewClose = document.querySelector("[data-action='close-overview']");
  const notesPanel = document.querySelector(".notes-panel");
  const notesBody = document.querySelector(".notes-body");
  let index = 0;

  if ("scrollRestoration" in history) history.scrollRestoration = "manual";

  function hashIndex() {
    const match = window.location.hash.match(/slide-(\d+)/);
    if (!match) return 0;
    return Math.max(0, Math.min(slides.length - 1, Number(match[1]) - 1));
  }

  function updateHash() {
    const hash = `#slide-${index + 1}`;
    if (window.location.hash !== hash) history.replaceState(null, "", hash);
  }

  function showSlide(nextIndex, options = {}) {
    index = Math.max(0, Math.min(slides.length - 1, nextIndex));
    slides.forEach((slide, i) => {
      const active = i === index;
      slide.classList.toggle("active", active);
      slide.setAttribute("aria-hidden", String(!active));
    });
    progress.style.width = `${((index + 1) / slides.length) * 100}%`;
    counter.textContent = `${String(index + 1).padStart(2, "0")} / ${String(slides.length).padStart(2, "0")}`;
    prevButton.disabled = index === 0;
    nextButton.disabled = index === slides.length - 1;
    if (!options.skipHash) updateHash();
    updateNotes();
    if (window.matchMedia("(max-width: 720px)").matches) {
      window.scrollTo(0, 0);
      requestAnimationFrame(() => window.scrollTo(0, 0));
    }
    document.title = `${slides[index].dataset.short || "JEV Explained"} · ${index + 1}/${slides.length}`;
  }

  function move(delta) { showSlide(index + delta); }

  function fitSlide() {
    if (window.matchMedia("(max-width: 720px)").matches) {
      document.documentElement.style.setProperty("--slide-scale", "1");
      return;
    }
    const width = stage.clientWidth - 22;
    const height = stage.clientHeight - 22;
    const scale = Math.min(width / 1600, height / 900);
    document.documentElement.style.setProperty("--slide-scale", String(Math.max(0.35, scale)));
  }

  function buildOverview() {
    overviewGrid.replaceChildren();
    slides.forEach((slide, i) => {
      const button = document.createElement("button");
      button.className = "overview-card";
      button.type = "button";
      button.innerHTML = `<span class="n">${String(i + 1).padStart(2, "0")}</span><span class="t"></span>`;
      button.querySelector(".t").textContent = slide.dataset.short || slide.querySelector("h1,h2")?.textContent || `Slide ${i + 1}`;
      button.addEventListener("click", () => {
        showSlide(i);
        closeOverview();
      });
      overviewGrid.appendChild(button);
    });
  }

  function openOverview() {
    overview.hidden = false;
    overview.setAttribute("aria-hidden", "false");
    overviewGrid.querySelectorAll(".overview-card")[index]?.focus();
  }

  function closeOverview() {
    overview.hidden = true;
    overview.setAttribute("aria-hidden", "true");
    overviewButton.focus();
  }

  function updateNotes() {
    if (notesPanel.hidden) return;
    const notes = slides[index].querySelector(".presenter-notes");
    notesBody.lang = notes?.lang || "";
    notesBody.innerHTML = notes ? notes.innerHTML : "<p>No presenter notes for this slide.</p>";
  }

  function toggleNotes() {
    notesPanel.hidden = !notesPanel.hidden;
    notesPanel.setAttribute("aria-hidden", String(notesPanel.hidden));
    notesButton.setAttribute("aria-pressed", String(!notesPanel.hidden));
    updateNotes();
  }

  async function toggleFullscreen() {
    try {
      if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
      else await document.exitFullscreen();
    } catch (error) {
      console.warn("Fullscreen is unavailable in this browser context.", error);
    }
  }

  function isTypingTarget(target) {
    return target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      target?.isContentEditable;
  }

  document.addEventListener("keydown", (event) => {
    if (isTypingTarget(event.target)) return;
    if (!overview.hidden && event.key === "Escape") {
      closeOverview();
      return;
    }
    switch (event.key) {
      case "ArrowRight":
      case "PageDown":
      case " ":
        event.preventDefault(); move(1); break;
      case "ArrowLeft":
      case "PageUp":
        event.preventDefault(); move(-1); break;
      case "Home":
        event.preventDefault(); showSlide(0); break;
      case "End":
        event.preventDefault(); showSlide(slides.length - 1); break;
      case "o":
      case "O":
        event.preventDefault(); overview.hidden ? openOverview() : closeOverview(); break;
      case "f":
      case "F":
        event.preventDefault(); toggleFullscreen(); break;
      case "n":
      case "N":
        event.preventDefault(); toggleNotes(); break;
      case "Escape":
        if (!notesPanel.hidden) toggleNotes();
        break;
    }
  });

  prevButton.addEventListener("click", () => move(-1));
  nextButton.addEventListener("click", () => move(1));
  overviewButton.addEventListener("click", openOverview);
  overviewClose.addEventListener("click", closeOverview);
  fullscreenButton.addEventListener("click", toggleFullscreen);
  notesButton.addEventListener("click", toggleNotes);
  printButton.addEventListener("click", () => window.print());
  window.addEventListener("resize", fitSlide);
  window.addEventListener("hashchange", () => showSlide(hashIndex(), { skipHash: true }));

  document.addEventListener("fullscreenchange", () => {
    fullscreenButton.setAttribute("aria-pressed", String(Boolean(document.fullscreenElement)));
    fitSlide();
  });

  buildOverview();
  showSlide(hashIndex(), { skipHash: true });
  fitSlide();
})();
