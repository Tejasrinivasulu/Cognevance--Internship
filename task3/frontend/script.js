/* ============================================
   Customer Churn Prediction API — Scripts
   Sticky navbar, smooth scroll, animations
   ============================================ */

(function () {
  "use strict";

  const navbar = document.getElementById("navbar");
  const navToggle = document.getElementById("navToggle");
  const navLinks = document.getElementById("navLinks");
  const backTop = document.getElementById("backTop");
  const yearEl = document.getElementById("year");

  // Current year in footer
  if (yearEl) {
    yearEl.textContent = String(new Date().getFullYear());
  }

  // Sticky / glass navbar on scroll
  function onScroll() {
    const y = window.scrollY || document.documentElement.scrollTop;
    if (navbar) {
      navbar.classList.toggle("scrolled", y > 24);
    }
    if (backTop) {
      backTop.classList.toggle("show", y > 420);
    }
    highlightNav();
  }

  // Mobile menu toggle
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
      navLinks.classList.toggle("open");
      const icon = navToggle.querySelector("i");
      if (icon) {
        icon.classList.toggle("fa-bars");
        icon.classList.toggle("fa-xmark");
      }
    });

    navLinks.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        navLinks.classList.remove("open");
        const icon = navToggle.querySelector("i");
        if (icon) {
          icon.classList.add("fa-bars");
          icon.classList.remove("fa-xmark");
        }
      });
    });
  }

  // Smooth scroll for same-page anchors
  document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener("click", function (e) {
      const id = anchor.getAttribute("href");
      if (!id || id === "#") return;
      const target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      const offset = (navbar ? navbar.offsetHeight : 72) + 8;
      const top = target.getBoundingClientRect().top + window.pageYOffset - offset;
      window.scrollTo({ top: top, behavior: "smooth" });
    });
  });

  // Active nav link based on scroll position
  function highlightNav() {
    const sections = ["home", "about", "features"];
    const scrollPos = window.scrollY + 120;
    let current = "home";

    sections.forEach(function (id) {
      const el = document.getElementById(id);
      if (el && el.offsetTop <= scrollPos) {
        current = id;
      }
    });

    document.querySelectorAll(".nav-links a[data-nav]").forEach(function (link) {
      const nav = link.getAttribute("data-nav");
      link.classList.toggle("active", nav === current);
    });
  }

  // Scroll reveal animations
  const revealEls = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );
    revealEls.forEach(function (el) {
      io.observe(el);
    });
  } else {
    revealEls.forEach(function (el) {
      el.classList.add("visible");
    });
  }

  // Back to top
  if (backTop) {
    backTop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  // Live API status (optional polish) — badge removed from landing

  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
})();
