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
    this.githubLink.href = item.url || '#';

    this.contentEl.innerHTML = `
      <div class="modal-loading">
        <i class="icon" data-icon="loader-2"></i>
        <p>Loading content...</p>
      </div>
    `;
    Icons.replace();

    if (item.path) {
      try {
        const rawUrl = item.url
          .replace('github.com', 'raw.githubusercontent.com')
          .replace('/blob/', '/');
        const response = await fetch(rawUrl);
        if (!response.ok) throw new Error('Failed to fetch');
        const content = await response.text();
        this.displayContent(content, item);
      } catch (error) {
        this.contentEl.innerHTML = `
          <div class="error" style="margin: 24px;">
            <i class="icon" data-icon="alert-triangle"></i>
            <p>Failed to load content: ${error.message}</p>
            <a href="${item.url}" target="_blank" rel="noopener" class="btn-primary" style="margin-top:16px;display:inline-flex;">
              <i class="icon" data-icon="external-link"></i> View on GitHub
            </a>
          </div>
        `;
        Icons.replace();
      }
    }
  },

  displayContent(content, item) {
    const isMD = item.name.endsWith('.md') || item.name.endsWith('.prompt.md');
    const isPy = item.name.endsWith('.py');
    const isYml = item.name.endsWith('.yaml') || item.name.endsWith('.yml');

    let html;
    if (isMD) {
      html = this.renderMarkdown(content);
    } else if (isPy || isYml) {
      html = this.renderCode(content, isPy ? 'python' : 'yaml');
    } else {
      html = this.renderCode(content, 'text');
    }
    this.contentEl.innerHTML = `<div class="content-viewer ${isMD ? 'markdown' : ''}">${html}</div>`;
    Icons.replace();
  },

  renderMarkdown(text) {
    let html = text
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/^### (.*$)/gm,'<h3>$1</h3>')
      .replace(/^## (.*$)/gm,'<h2>$1</h2>')
      .replace(/^# (.*$)/gm,'<h1>$1</h1>')
      .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
      .replace(/\*(.*?)\*/g,'<em>$1</em>')
      .replace(/```(\w*)\n([\s\S]*?)```/g,'<pre><code class="language-$1">$2</code></pre>')
      .replace(/`([^`]+)`/g,'<code>$1</code>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>')
      .replace(/^- (.*$)/gm,'<li>$1</li>')
      .replace(/^\d+\. (.*$)/gm,'<li>$1</li>')
      .replace(/\n\n/g,'</p><p>')
      .replace(/\n/g,'<br>');
    html = html.replace(/(<li>.*<\/li>)/gs,'<ul>$1</ul>').replace(/<\/ul>\s*<ul>/g,'');
    return `<p>${html}</p>`;
  },

  renderCode(code, language) {
    const esc = code.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    return `<pre><code class="language-${language}">${esc}</code></pre>`;
  },

  close() {
    this.modal.classList.remove('active');
    document.body.style.overflow = '';
    this.contentEl.innerHTML = '';
  }
};

// Initialize app with STATIC data (no GitHub API calls!)
document.addEventListener('DOMContentLoaded', async () => {
  console.log('LLM-Whisperer Website Initializing...');

  UI.initTheme();
  UI.initMobileMenu();
  UI.initHeader();
  UI.initNavigation();
  Modal.init();

  UI.showLoading('skills-grid');
  UI.showLoading('sample-code-grid');
  UI.showLoading('scripts-grid');
  UI.showLoading('pipelines-grid');

  try {
    // Load static data.json (pre-generated, no API rate limits!)
    const response = await fetch('js/data.json');
    const raw = await response.json();

    allSkills = processSkills(raw.skills);
    allSampleCode = processSampleCode(raw.sample_code);
    allScripts = processScripts(raw.scripts);
    allPipelines = processPipelines(raw.pipelines);
    allNotebooks = processNotebooks(raw.notebooks);
    allConfigs = processConfigs(raw.configs);

    UI.updateStats({
      skills: allSkills.length,
      code: allSampleCode.length,
      scripts: allScripts.length,
      stars: 0,
    });

    allItems = [...allSkills, ...allSampleCode, ...allScripts, ...allPipelines, ...allNotebooks, ...allConfigs];

    renderSkills();
    renderSampleCode();
    renderScripts();
    renderPipelines();
    renderNotebooks();
    renderConfigs();
    renderDocs();

    UI.initSearch(allItems);

    document.addEventListener('item-click', (e) => {
      Modal.open(e.detail);
    });

    Icons.replace();
    console.log('LLM-Whisperer Website Loaded Successfully!');
  } catch (error) {
    console.error('Failed to initialize:', error);
    UI.showError('skills-grid', `Failed to load: ${error.message}`);
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
        url: `${GH}/tree/main/skills/${skills[0].categorySlug || catName.toLowerCase()}`,
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
      <a href="https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills" target="_blank" rel="noopener" class="btn-primary">
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
      url: `https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/blob/main/README.md`,
      icon: 'file-text',
      category: 'Documentation',
    },
    {
      name: 'CONTRIBUTING',
      url: `https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/blob/main/CONTRIBUTING.md`,
      icon: 'git-pull-request',
      category: 'Documentation',
    },
    {
      name: 'Architecture Docs',
      url: `https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/docs/architecture`,
      icon: 'building',
      category: 'Documentation',
    },
    {
      name: 'Guides',
      url: `https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/docs/guides`,
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
