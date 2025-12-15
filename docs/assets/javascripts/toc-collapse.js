// Enhance collapsible behavior for primary (left) navigation in Material for MkDocs
// Reads expansion depth configuration from mkdocs.yml
(function () {
  // Read expand depth from mkdocs config (set in extra.nav_config.expand_depth)
  // Falls back to 1 if not configured
  var DEFAULT_EXPAND_DEPTH = window.nav_config?.expand_depth ?? 1;

  function makeAccordion(containerSelector) {
    var container = document.querySelector(containerSelector);
    if (!container) return;

    // Avoid duplicating toggles on SPA navigations
    container.querySelectorAll('.toc-accordion-toggle').forEach(function (btn) {
      btn.remove();
    });

    // Remove old control buttons
    var oldControls = container.querySelector('.toc-control-buttons');
    if (oldControls) oldControls.remove();

    // Recursively process all navigation items at all levels
    function processNavItems(parentList, depth) {
      if (!parentList) return;
      depth = depth || 0;

      var items = parentList.querySelectorAll(':scope > .md-nav__item');
      items.forEach(function (item) {
        var sublist = item.querySelector(':scope > .md-nav__list');
        var link = item.querySelector(':scope > .md-nav__link');

        if (sublist) {
          // Check if this item or any descendant is marked active
          var isActiveBranch = item.classList.contains('md-nav__item--active');
          var hasActiveChild = !!item.querySelector('.md-nav__item--active');
          var shouldBeOpen = isActiveBranch || hasActiveChild || depth < DEFAULT_EXPAND_DEPTH;

          // Set initial state
          if (!shouldBeOpen) {
            sublist.style.display = 'none';
            item.classList.add('toc-collapsed');
          } else {
            item.classList.remove('toc-collapsed');
          }

          // Inject a toggle button before the link
          var btn = document.createElement('button');
          btn.className = 'toc-accordion-toggle';
          btn.setAttribute('aria-expanded', shouldBeOpen ? 'true' : 'false');
          btn.setAttribute('data-depth', depth);
          btn.title = shouldBeOpen ? 'Collapse section' : 'Expand section';
          btn.textContent = shouldBeOpen ? '−' : '+';
          if (link && link.parentNode) {
            link.parentNode.insertBefore(btn, link);
          }

          // Toggle handler
          btn.addEventListener('click', function (ev) {
            ev.stopPropagation();
            var isOpen = sublist.style.display !== 'none';

            // Toggle current
            var nextOpen = !isOpen;
            sublist.style.display = nextOpen ? 'block' : 'none';
            btn.setAttribute('aria-expanded', nextOpen ? 'true' : 'false');
            btn.title = nextOpen ? 'Collapse section' : 'Expand section';
            btn.textContent = nextOpen ? '−' : '+';
            item.classList.toggle('toc-collapsed', !nextOpen);
          });

          // Optional: clicking the link toggles the group if it points to a heading within the page or is a section stub
          if (link) {
            link.addEventListener('click', function (ev) {
              // Only intercept if the link is an anchor to current page or a stub (no href or '#')
              var href = link.getAttribute('href');
              var isLocalAnchor = href && href.startsWith('#');
              var isStub = !href || href === '#';
              if (isLocalAnchor || isStub) {
                ev.preventDefault();
                btn.click();
              }
            });
          }

          // Recursively process nested items
          processNavItems(sublist, depth + 1);
        }
      });
    }

    var rootList = container.querySelector(':scope > .md-nav__list');
    if (rootList) {
      processNavItems(rootList, 0);
    }
  }


  function scheduleInit() {
    // Initialize primary navigation (left sidebar) with full recursive collapsibility
    // The right TOC panel is now integrated into the left sidebar via toc.integrate
    makeAccordion('.md-nav--primary');

    // Re-init on SPA navigation: observe changes to main content area
    var container = document.querySelector('main');
    if (!container) return;
    var observer = new MutationObserver(function () {
      makeAccordion('.md-nav--primary');
    });
    observer.observe(container, { childList: true, subtree: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', scheduleInit);
  } else {
    scheduleInit();
  }
})();
