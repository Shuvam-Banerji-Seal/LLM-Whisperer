/* ============================================
   LLM-Whisperer - Main Application
   ============================================ */

// Global state
let allItems = [];
let allSkills = [];
let allSampleCode = [];
let allScripts = [];
let allPipelines = [];
let allNotebooks = [];
let allConfigs = [];

// Content Viewer Modal
const Modal = {
  modal: null,
  contentEl: null,
  titleEl: null,
  subtitleEl: null,
  iconEl: null,
  githubLink: null,

  init() {
    this.modal = document.getElementById('content-modal');
    this.contentEl = document.getElementById('modal-content');
    this.titleEl = document.getElementById('modal-title');
    this.subtitleEl = document.getElementById('modal-subtitle');
    this.iconEl = document.getElementById('modal-icon');
    this.githubLink = document.getElementById('modal-github-link');

    document.getElementById('modal-close').addEventListener('click', () => this.close());
    this.modal.addEventListener('click', (e) => {
      if (e.target === this.modal) this.close();
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.modal.classList.contains('active')) {
        this.close();
      }
    });
  },

  async open(item) {
    this.modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    this.titleEl.textContent = item.name || 'Loading...';
    this.subtitleEl.textContent = item.category || '';
    this.iconEl.setAttribute('data-icon', item.icon || 'file');
    this.githubLink.href = item.url || `https://github.com/${API.repo}/tree/main/${item.path}`;

    this.contentEl.innerHTML = `
      <div class="modal-loading">
        <i class="icon" data-icon="loader-2"></i>
        <p>Loading content...</p>
      </div>
    `;
    Icons.replace();

    if (item.path) {
      try {
        const content = await this.fetchContent(item);
        this.displayContent(content, item);
      } catch (error) {
        this.contentEl.innerHTML = `
          <div class="error" style="margin: 24px;">
            <i class="icon" data-icon="alert-triangle"></i>
            <p>Failed to load content: ${error.message}</p>
            <a href="${item.url}" target="_blank" rel="noopener" class="btn-primary" style="margin-top: 16px; display: inline-flex;">
              View on GitHub
            </a>
          </div>
        `;
        Icons.replace();
      }
    } else if (item.url) {
      window.open(item.url, '_blank');
      this.close();
    }
  },

  async fetchContent(item) {
    const rawUrl = item.url
      .replace('github.com', 'raw.githubusercontent.com')
      .replace('/blob/', '/');

    const response = await fetch(rawUrl);
    if (!response.ok) throw new Error('Failed to fetch file');
    return await response.text();
  },

  displayContent(content, item) {
    const isMarkdown = item.name.endsWith('.md') || item.name.endsWith('.prompt.md');
    const isPython = item.name.endsWith('.py');
    const isYaml = item.name.endsWith('.yaml') || item.name.endsWith('.yml');

    let htmlContent;

    if (isMarkdown) {
      htmlContent = this.renderMarkdown(content);
    } else if (isPython || isYaml) {
      htmlContent = this.renderCode(content, isPython ? 'python' : 'yaml');
    } else {
      htmlContent = this.renderCode(content, 'text');
    }

    this.contentEl.innerHTML = `<div class="content-viewer ${isMarkdown ? 'markdown' : ''}">${htmlContent}</div>`;
    Icons.replace();
  },

  renderMarkdown(text) {
    // Simple markdown to HTML conversion
    let html = text
      // Escape HTML
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      // Headers
      .replace(/^### (.*$)/gm, '<h3>$1</h3>')
      .replace(/^## (.*$)/gm, '<h2>$1</h2>')
      .replace(/^# (.*$)/gm, '<h1>$1</h1>')
      // Bold and italic
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      // Code blocks
      .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
      // Inline code
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      // Links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
      // Lists
      .replace(/^\- (.*$)/gm, '<li>$1</li>')
      .replace(/^\d+\. (.*$)/gm, '<li>$1</li>')
      // Paragraphs
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br>');

    // Wrap list items
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    return `<p>${html}</p>`;
  },

  renderCode(code, language) {
    const escaped = code
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    return `<pre><code class="language-${language}">${escaped}</code></pre>`;
  },

  close() {
    this.modal.classList.remove('active');
    document.body.style.overflow = '';
    this.contentEl.innerHTML = '';
  }
};

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
  console.log('LLM-Whisperer Website Initializing...');

  // Initialize UI features
  UI.initTheme();
  UI.initMobileMenu();
  UI.initHeader();
  UI.initNavigation();
  Modal.init();

  // Show loading states
  UI.showLoading('skills-grid');
  UI.showLoading('sample-code-grid');
  UI.showLoading('scripts-grid');
  UI.showLoading('pipelines-grid');

  try {
    // Load all data in parallel
    const [skills, sampleCode, scripts, pipelines, repoInfo] = await Promise.all([
      API.getSkills(),
      API.getSampleCode(),
      API.getScripts(),
      API.getPipelines(),
      API.getRepoInfo().catch(() => null),
    ]);

    allSkills = skills;
    allSampleCode = sampleCode;
    allScripts = scripts;
    allPipelines = pipelines;

    // Update stats
    if (repoInfo) {
      UI.updateStats({
        skills: skills.length,
        code: sampleCode.length,
        scripts: scripts.length,
        stars: repoInfo.stargazers_count || 0,
      });
    }

    // Get notebooks
    try {
      const notebooks = await API.getContents('notebooks');
      allNotebooks = notebooks
        .filter((f) => f.type === 'dir')
        .map((dir) => ({
          name: dir.name,
          path: dir.path,
          url: dir.html_url,
          type: 'notebook',
          icon: 'notebook',
          category: 'Notebooks',
        }));
    } catch (e) {
      console.warn('Failed to load notebooks:', e);
    }

    // Get configs
    try {
      const configs = await API.getContents('configs');
      allConfigs = configs
        .filter((f) => f.type === 'dir')
        .map((dir) => ({
          name: dir.name,
          path: dir.path,
          url: dir.html_url,
          type: 'config',
          icon: 'settings',
          category: 'Configs',
        }));
    } catch (e) {
      console.warn('Failed to load configs:', e);
    }

    // Combine all items for search
    allItems = [
      ...allSkills,
      ...allSampleCode,
      ...allScripts,
      ...allPipelines,
      ...allNotebooks,
      ...allConfigs,
    ];

    // Render all sections
    renderSkills();
    renderSampleCode();
    renderScripts();
    renderPipelines();
    renderNotebooks();
    renderConfigs();
    renderDocs();

// Initialize search
  UI.initSearch(allItems);

  // Add click handlers for item cards (delegated to document)
  document.addEventListener('item-click', (e) => {
    Modal.open(e.detail);
  });

  // Initialize Lucide icons
  Icons.replace();

    console.log('LLM-Whisperer Website Loaded Successfully!');
  } catch (error) {
    console.error('Failed to initialize app:', error);
    UI.showError('skills-grid', `Failed to load data: ${error.message}`);
  }
});

/**
 * Render Skills Section
 */
function renderSkills() {
  const container = document.getElementById('skills-grid');
  if (!container) return;

  // Group skills by category
  const categories = {};
  allSkills.forEach((skill) => {
    const cat = skill.category || 'Other';
    if (!categories[cat]) {
      categories[cat] = [];
    }
    categories[cat].push(skill);
  });

  // Create category cards
  const fragment = document.createDocumentFragment();
  Object.entries(categories).forEach(([catName, skills], index) => {
    const card = document.createElement('div');
    card.className = 'category-card fade-in';
    card.style.animationDelay = `${index * 50}ms`;
    card.onclick = () => {
      Modal.open({
        name: catName + ' Skills',
        category: 'Skills',
        icon: skills[0].icon || 'folder',
        url: `https://github.com/${API.repo}/tree/main/skills/${skills[0].categorySlug || catName.toLowerCase()}`,
        path: `skills/${skills[0].categorySlug || catName.toLowerCase()}`
      });
    };

    card.innerHTML = `
      <i class="icon category-icon" data-icon="${skills[0].icon || 'folder'}"></i>
      <h4>${catName}</h4>
      <p>${skills.length} skill${skills.length !== 1 ? 's' : ''} available</p>
      <span class="category-count">
        <i class="icon" data-icon="file-text"></i>
        ${skills.length} items
      </span>
    `;

    fragment.appendChild(card);
  });

  container.innerHTML = '';
  container.className = 'grid grid-3';
  container.appendChild(fragment);

  // Show individual skills button
  const section = document.getElementById('skills');
  if (section) {
    const buttonContainer = document.createElement('div');
    buttonContainer.style.textAlign = 'center';
    buttonContainer.style.marginTop = '32px';
    buttonContainer.innerHTML = `
      <a href="https://github.com/${API.repo}/tree/main/skills" target="_blank" rel="noopener" class="btn-primary">
        <i class="icon" data-icon="external-link"></i>
        View All Skills on GitHub
      </a>
    `;
    section.querySelector('.section:last-child')?.appendChild(buttonContainer);
  }

  Icons.replace();
}

/**
 * Render Sample Code Section
 */
function renderSampleCode() {
  UI.renderGrid('sample-code-grid', allSampleCode, 3);
}

/**
 * Render Scripts Section
 */
function renderScripts() {
  UI.renderGrid('scripts-grid', allScripts, 3);
}

/**
 * Render Pipelines Section
 */
function renderPipelines() {
  const container = document.getElementById('pipelines-grid');
  if (!container) return;

  container.innerHTML = '';
  container.className = 'grid grid-4';

  allPipelines.forEach((pipe, index) => {
    const card = document.createElement('div');
    card.className = 'item-card fade-in';
    card.style.animationDelay = `${index * 50}ms`;
    card.onclick = () => {
      Modal.open(pipe);
    };

    card.innerHTML = `
      <div class="item-header">
        <i class="icon" data-icon="${pipe.icon || 'git-branch'}"></i>
        <div>
          <a href="${pipe.url}" target="_blank" rel="noopener" class="item-title">
            ${pipe.name}
          </a>
        </div>
      </div>
      <div class="item-meta">
        <span class="badge"><i class="icon" data-icon="git-branch"></i> Pipeline</span>
        ${pipe.hasConfigs ? '<span class="badge"><i class="icon" data-icon="settings"></i> Configs</span>' : ''}
      </div>
    `;

    container.appendChild(card);
  });

  Icons.replace();
}

/**
 * Render Notebooks Section
 */
function renderNotebooks() {
  if (allNotebooks.length === 0) {
    const section = document.getElementById('notebooks');
    if (section) section.style.display = 'none';
    return;
  }
  UI.renderGrid('notebooks-grid', allNotebooks, 3);
}

/**
 * Render Configs Section
 */
function renderConfigs() {
  if (allConfigs.length === 0) {
    const section = document.getElementById('configs');
    if (section) section.style.display = 'none';
    return;
  }
  UI.renderGrid('configs-grid', allConfigs, 4);
}

/**
 * Render Docs Section
 */
function renderDocs() {
  const docs = [
    {
      name: 'README',
      url: `https://github.com/${API.repo}/blob/main/README.md`,
      icon: 'file-text',
      category: 'Documentation',
    },
    {
      name: 'CONTRIBUTING',
      url: `https://github.com/${API.repo}/blob/main/CONTRIBUTING.md`,
      icon: 'git-pull-request',
      category: 'Documentation',
    },
    {
      name: 'Architecture Docs',
      url: `https://github.com/${API.repo}/tree/main/docs/architecture`,
      icon: 'building',
      category: 'Documentation',
    },
    {
      name: 'Guides',
      url: `https://github.com/${API.repo}/tree/main/docs/guides`,
      icon: 'book-open',
      category: 'Documentation',
    },
  ];

  UI.renderGrid('docs-grid', docs, 4);
}

// Add btn-primary style dynamically
const style = document.createElement('style');
style.textContent = `
  .btn-primary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 12px 24px;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    color: white;
    border-radius: var(--border-radius-sm);
    font-weight: 600;
    transition: all var(--transition-fast);
    border: none;
    cursor: pointer;
  }

  .btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    color: white;
  }
`;
document.head.appendChild(style);
