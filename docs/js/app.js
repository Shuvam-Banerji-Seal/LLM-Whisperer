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

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
  console.log('LLM-Whisperer Website Initializing...');

  // Initialize UI features
  UI.initTheme();
  UI.initMobileMenu();
  UI.initHeader();
  UI.initNavigation();

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

    // Initialize Lucide icons
    if (window.lucide) {
      lucide.createIcons();
    }

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
      window.open(`https://github.com/${API.repo}/tree/main/skills/${skills[0].categorySlug || catName.toLowerCase()}`, '_blank');
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

  if (window.lucide) lucide.createIcons();
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
      window.open(pipe.url, '_blank');
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

  if (window.lucide) lucide.createIcons();
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
