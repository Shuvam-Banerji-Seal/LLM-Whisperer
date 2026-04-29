/* ============================================
   LLM-Whisperer - UI Module
   ============================================ */

const UI = {
  /**
   * Render a grid of items (skills, code, scripts, etc.)
   */
  renderGrid(containerId, items, columns = 3) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';
    container.className = `grid grid-${columns}`;

    if (items.length === 0) {
      container.innerHTML = `
        <div class="empty" style="grid-column: 1 / -1;">
          <i class="icon" data-icon="inbox"></i>
          <p>No items found</p>
        </div>
      `;
      return;
    }

    items.forEach((item, index) => {
      const card = this.createItemCard(item, index);
      container.appendChild(card);
    });

    // Initialize Lucide icons
    if (window.lucide) {
      lucide.createIcons();
    }
  },

  /**
   * Create an item card element
   */
  createItemCard(item, index) {
    const card = document.createElement('div');
    card.className = 'item-card fade-in';
    card.style.animationDelay = `${index * 50}ms`;

    const icon = item.icon || 'file-text';
    const url = item.url || `https://github.com/${API.repo}/tree/main/${item.path}`;

    card.innerHTML = `
      <div class="item-header">
        <i class="icon" data-icon="${icon}"></i>
        <a href="${url}" target="_blank" rel="noopener" class="item-title">
          ${this.escapeHtml(item.name)}
        </a>
      </div>
      ${item.description ? `<p class="item-description">${this.escapeHtml(item.description)}</p>` : ''}
      <div class="item-meta">
        ${item.category ? `<span class="badge"><i class="icon" data-icon="folder"></i> ${this.escapeHtml(item.category)}</span>` : ''}
        ${item.type ? `<span class="badge"><i class="icon" data-icon="${this.getTypeIcon(item.type)}"></i> ${this.escapeHtml(item.type)}</span>` : ''}
        ${item.language ? `<span class="badge"><i class="icon" data-icon="code-2"></i> ${this.escapeHtml(item.language)}</span>` : ''}
      </div>
    `;

    return card;
  },

  /**
   * Render category cards
   */
  renderCategories(containerId, categories) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';
    container.className = 'grid grid-3';

    categories.forEach((cat, index) => {
      const card = document.createElement('div');
      card.className = 'category-card fade-in';
      card.style.animationDelay = `${index * 50}ms`;
      card.onclick = () => {
        window.location.hash = cat.slug || cat.name.toLowerCase().replace(/ /g, '-');
      };

      card.innerHTML = `
        <i class="icon category-icon" data-icon="${cat.icon || 'folder'}"></i>
        <h4>${this.escapeHtml(cat.name)}</h4>
        <p>${this.escapeHtml(cat.description || `Browse ${cat.name} collection`)}</p>
        <span class="category-count">
          <i class="icon" data-icon="file-text"></i>
          ${cat.count || 0} items
        </span>
      `;

      container.appendChild(card);
    });

    if (window.lucide) {
      lucide.createIcons();
    }
  },

  /**
   * Render search results
   */
  renderSearchResults(query, allItems) {
    const resultsContainer = document.getElementById('search-results');
    const resultsGrid = document.getElementById('search-results-grid');
    const resultsCount = document.getElementById('search-results-count');

    if (!resultsContainer || !resultsGrid) return;

    if (!query.trim()) {
      resultsContainer.classList.remove('active');
      return;
    }

    const filtered = allItems.filter((item) => {
      const searchStr = JSON.stringify(item).toLowerCase();
      return searchStr.includes(query.toLowerCase());
    });

    resultsCount.textContent = `Found ${filtered.length} result${filtered.length !== 1 ? 's' : ''} for "${query}"`;
    this.renderGrid('search-results-grid', filtered, 3);
    resultsContainer.classList.add('active');

    // Hide other sections
    document.querySelectorAll('section[id]').forEach((section) => {
      section.style.display = 'none';
    });
  },

  /**
   * Show all sections (when search is cleared)
   */
  showAllSections() {
    document.querySelectorAll('section[id]').forEach((section) => {
      section.style.display = '';
    });
    const resultsContainer = document.getElementById('search-results');
    if (resultsContainer) {
      resultsContainer.classList.remove('active');
    }
  },

  /**
   * Update stats in hero section
   */
  updateStats(stats) {
    const elements = {
      'stat-skills': stats.skills || 0,
      'stat-code': stats.code || 0,
      'stat-scripts': stats.scripts || 0,
      'stat-stars': stats.stars || 0,
    };

    for (const [id, value] of Object.entries(elements)) {
      const el = document.getElementById(id);
      if (el) el.textContent = value.toLocaleString();
    }
  },

  /**
   * Show loading state
   */
  showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
      <div class="loading" style="grid-column: 1 / -1;">
        <i class="icon" data-icon="loader-2"></i>
        <p>Loading...</p>
      </div>
    `;
    if (window.lucide) lucide.createIcons();
  },

  /**
   * Show error state
   */
  showError(containerId, message) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
      <div class="error" style="grid-column: 1 / -1;">
        <i class="icon" data-icon="alert-triangle"></i>
        <p>${this.escapeHtml(message)}</p>
      </div>
    `;
    if (window.lucide) lucide.createIcons();
  },

  /**
   * Helper: Escape HTML
   */
  escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  },

  /**
   * Helper: Get icon for type
   */
  getTypeIcon(type) {
    const icons = {
      skill: 'brain',
      'sample-code': 'code-2',
      script: 'terminal',
      pipeline: 'git-branch',
      notebook: 'notebook',
      config: 'settings',
      doc: 'book-open',
    };
    return icons[type] || 'file-text';
  },

  /**
   * Initialize theme toggle
   */
  initTheme() {
    const toggle = document.getElementById('theme-toggle');
    if (!toggle) return;

    // Check saved theme or system preference
    const savedTheme = localStorage.getItem('theme');
    const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme) {
      document.documentElement.setAttribute('data-theme', savedTheme);
    } else if (systemDark) {
      document.documentElement.setAttribute('data-theme', 'dark');
    }

    toggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
    });
  },

  /**
   * Initialize search
   */
  initSearch(allItems) {
    const input = document.getElementById('search-input');
    if (!input) return;

    let debounceTimer;

    input.addEventListener('input', (e) => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        this.renderSearchResults(e.target.value, allItems);
      }, 300);
    });

    // Keyboard shortcut (Cmd/Ctrl + K)
    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        input.focus();
      }
      if (e.key === 'Escape' && document.activeElement === input) {
        input.blur();
        input.value = '';
        this.showAllSections();
      }
    });
  },

  /**
   * Initialize mobile menu
   */
  initMobileMenu() {
    const toggle = document.getElementById('mobile-menu-toggle');
    const mobileNav = document.getElementById('mobile-nav');

    if (!toggle || !mobileNav) return;

    toggle.addEventListener('click', () => {
      mobileNav.classList.toggle('active');
    });

    // Close on link click
    mobileNav.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        mobileNav.classList.remove('active');
      });
    });
  },

  /**
   * Initialize header scroll effect
   */
  initHeader() {
    const header = document.getElementById('header');
    if (!header) return;

    let lastScroll = 0;

    window.addEventListener('scroll', () => {
      const currentScroll = window.pageYOffset;

      if (currentScroll > 50) {
        header.classList.add('scrolled');
      } else {
        header.classList.remove('scrolled');
      }

      lastScroll = currentScroll;
    });
  },

  /**
   * Initialize navigation active state
   */
  initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link, .mobile-nav-link');

    navLinks.forEach((link) => {
      link.addEventListener('click', (e) => {
        navLinks.forEach((l) => l.classList.remove('active'));
        link.classList.add('active');

        // Close mobile nav if open
        const mobileNav = document.getElementById('mobile-nav');
        if (mobileNav) mobileNav.classList.remove('active');
      });
    });
  },
};
