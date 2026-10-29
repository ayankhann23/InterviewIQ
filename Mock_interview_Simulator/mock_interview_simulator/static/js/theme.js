/**
 * Theme Manager for InterviewIQ Dashboard & Simulator
 * Handles Dark / Light mode toggling with persistence in localStorage.
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'interviewiq_theme';

  function getStoredTheme() {
    try {
      var saved = localStorage.getItem(STORAGE_KEY);
      if (saved === 'light' || saved === 'dark') {
        return saved;
      }
    } catch (e) {
      // localStorage may fail in restricted contexts
    }
    return 'dark'; // Default theme
  }

  function applyTheme(theme) {
    var validTheme = theme === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', validTheme);

    try {
      localStorage.setItem(STORAGE_KEY, validTheme);
    } catch (e) {}

    var darkBtn = document.getElementById('theme-btn-dark');
    var lightBtn = document.getElementById('theme-btn-light');

    if (darkBtn && lightBtn) {
      if (validTheme === 'light') {
        darkBtn.classList.remove('active');
        darkBtn.setAttribute('aria-checked', 'false');
        lightBtn.classList.add('active');
        lightBtn.setAttribute('aria-checked', 'true');
      } else {
        lightBtn.classList.remove('active');
        lightBtn.setAttribute('aria-checked', 'false');
        darkBtn.classList.add('active');
        darkBtn.setAttribute('aria-checked', 'true');
      }
    }
  }

  function initTheme() {
    var currentTheme = document.documentElement.getAttribute('data-theme') || getStoredTheme();
    applyTheme(currentTheme);

    var darkBtn = document.getElementById('theme-btn-dark');
    var lightBtn = document.getElementById('theme-btn-light');

    if (darkBtn) {
      darkBtn.addEventListener('click', function () {
        applyTheme('dark');
      });
    }

    if (lightBtn) {
      lightBtn.addEventListener('click', function () {
        applyTheme('light');
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTheme);
  } else {
    initTheme();
  }
})();
