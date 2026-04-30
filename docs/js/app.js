/* ============================================
   LLM-Whisperer v2 — Complete Application
   Static data, GitHub raw content, caching, lazy load
   ============================================ */

const REPO = 'Shuvam-Banerji-Seal/LLM-Whisperer';
const GH = 'https://github.com/' + REPO;
const RAW = 'https://raw.githubusercontent.com/' + REPO + '/main/';

// ============================================
// Icon Map (inline SVG strings)
// ============================================
const ICONS = {
  brain: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-1.54"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-1.54"/></svg>',
  terminal: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/></svg>',
  code: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
  gitBranch: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="6" x2="6" y1="3" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>',
  file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/></svg>',
  fileCode: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><path d="m10 13-2 2 2 2"/><path d="m14 17 2-2-2-2"/></svg>',
  fileText: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>',
  folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"/></svg>',
  database: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>',
  settings: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
  shield: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
  cpu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/></svg>',
  sparkles: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>',
  zap: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/></svg>',
  notebook: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 6h4"/><path d="M2 10h4"/><path d="M2 14h4"/><path d="M2 18h4"/><rect width="16" height="20" x="4" y="2" rx="2"/><path d="M16 2v20"/></svg>',
  book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>',
  users: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
  globe: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/></svg>',
  cloud: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>',
  image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>',
  clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
  box: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>',
  bot: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="18" height="10" x="3" y="11" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" x2="8" y1="16" y2="16"/><line x1="16" x2="16" y1="16" y2="16"/></svg>',
  filter: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
  clipboard: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="m9 14 2 2 4-4"/></svg>',
  trendingUp: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
  message: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
  minimize: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" x2="21" y1="10" y2="3"/><line x1="3" x2="10" y1="21" y2="14"/></svg>',
  video: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg>',
  x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>',
  download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>',
  external: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" x2="21" y1="14" y2="3"/></svg>',
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
  layers: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
  alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
};

const CAT_ICONS = {
  rag:'database', 'rag-advanced':'database', agentic:'users', agents:'bot',
  inference:'zap', 'fine-tuning':'settings', 'production-ops':'settings',
  safety:'shield', 'advanced-architectures':'cpu', 'advanced-reasoning':'brain',
  foundational:'book', 'llm-engineering':'code', 'data-preprocessing':'filter',
  datasets:'database', evaluation:'clipboard', templates:'layers',
  workflows:'gitBranch', fast:'zap', 'training-optimization':'trendingUp',
  'code-generation':'code', diffusion:'sparkles', huggingface:'heart',
  'image-generation':'image', 'infrastructure-deployment':'cloud',
  'knowledge-systems':'globe', 'long-context':'message',
  'model-merging':'gitBranch', moe:'layers', multimodal:'image',
  'prompt-engineering':'message', quantization:'minimize',
  'security-governance':'shield', 'specialized-ml-techniques':'cpu',
  'time-series':'clock', transformers:'box', turboquant:'zap',
  'video-generation':'video',
};

// ============================================
// State
// ============================================
let allItems = [];
let repoData = null;
const contentCache = new Map();

// ============================================
// Utilities
// ============================================
const $ = (sel, el=document) => el.querySelector(sel);
const $$ = (sel, el=document) => [...el.querySelectorAll(sel)];
const fmtName = s => s.split(/[-_]/).map(w=>w.charAt(0).toUpperCase()+w.slice(1)).join(' ');
const ghUrl = (path, type) => type === 'dir' ? `${GH}/tree/main/${path}` : `${GH}/blob/main/${path}`;
const rawUrl = path => `${RAW}${path}`;

function getIcon(name) { return ICONS[name] || ICONS.file; }

// ============================================
// Toast
// ============================================
function toast(message, type='success') {
  const container = $('#toastContainer');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `${type==='success' ? getIcon('check') : getIcon('alert')}<span>${message}</span>`;
  container.appendChild(el);
  setTimeout(() => {
    el.style.animation = 'toastOut 0.4s ease forwards';
    setTimeout(() => el.remove(), 400);
  }, 3000);
}

// ============================================
// Copy & Download
// ============================================
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast('Copied to clipboard!');
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    toast('Copied to clipboard!');
  }
}

function downloadFile(content, filename) {
  const blob = new Blob([content], {type: 'text/plain'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  toast(`Downloaded ${filename}`);
}

// ============================================
// Content Loading with Cache
// ============================================
async function loadContent(path) {
  if (contentCache.has(path)) return contentCache.get(path);
  try {
    const res = await fetch(rawUrl(path), { cache: 'force-cache' });
    if (!res.ok) throw new Error('Failed to fetch');
    const text = await res.text();
    contentCache.set(path, text);
    return text;
  } catch (e) {
    console.error('Content load error:', e);
    return null;
  }
}

// ============================================
// Markdown Renderer
// ============================================
function renderMarkdown(text) {
  let html = text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/^###### (.*$)/gm,'<h6>$1</h6>')
    .replace(/^##### (.*$)/gm,'<h5>$1</h5>')
    .replace(/^#### (.*$)/gm,'<h4>$1</h4>')
    .replace(/^### (.*$)/gm,'<h3>$1</h3>')
    .replace(/^## (.*$)/gm,'<h2>$1</h2>')
    .replace(/^# (.*$)/gm,'<h1>$1</h1>')
    .replace(/\*\*\*(.*?)\*\*\*/g,'<strong><em>$1</em></strong>')
    .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.*?)\*/g,'<em>$1</em>')
    .replace(/```(\w*)\n([\s\S]*?)```/g,'<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/^\- (.*$)/gm,'<li>$1</li>')
    .replace(/^\d+\. (.*$)/gm,'<li>$1</li>')
    .replace(/\n\n/g,'</p><p>')
    .replace(/\n/g,'<br>');

  // Fix lists
  let inList = false;
  html = html.split('</p><p>').map(p => {
    if (p.includes('<li>') && !inList) { inList = true; return '<ul>' + p; }
    if (!p.includes('<li>') && inList) { inList = false; return '</ul></p><p>' + p; }
    return p;
  }).join('</p><p>');
  if (inList) html += '</ul>';

  return `<div class="md-viewer">${html}</div>`;
}

function renderCode(text, lang) {
  const esc = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  return `<div class="code-viewer"><div class="code-header"><span class="code-lang">${lang}</span></div><div class="code-body"><pre><code>${esc}</code></pre></div></div>`;
}

// ============================================
// Modal
// ============================================
const Modal = {
  el: null,
  body: null,
  title: null,
  subtitle: null,
  icon: null,
  currentPath: null,
  currentContent: null,

  init() {
    this.el = $('#contentModal');
    this.body = $('#modalBody');
    this.title = $('#modalTitle');
    this.subtitle = $('#modalSubtitle');
    this.icon = $('#modalIcon');

    $('#modalCloseBtn').onclick = () => this.close();
    this.el.onclick = (e) => { if(e.target===this.el) this.close(); };
    $('#modalCopyBtn').onclick = () => {
      if(this.currentContent) copyToClipboard(this.currentContent);
    };
    $('#modalDownloadBtn').onclick = () => {
      if(this.currentContent && this.currentPath) {
        downloadFile(this.currentContent, this.currentPath.split('/').pop());
      }
    };
    document.addEventListener('keydown', (e) => {
      if(e.key==='Escape' && this.el.classList.contains('open')) this.close();
    });
  },

  async open(item) {
    // If this is a directory listing, show file browser
    if (item.isDir || !item.path) {
      this.openDir(item);
      return;
    }

    this.currentPath = item.path;
    this.currentContent = null;
    this.el.classList.add('open');
    document.body.style.overflow = 'hidden';

    this.title.textContent = item.name;
    this.subtitle.textContent = item.category || '';
    this.icon.innerHTML = getIcon(item.icon || 'file');
    $('#modalGithubBtn').href = item.url;
    $('#modalDownloadBtn').href = item.url;

    this.body.innerHTML = `<div class="modal-loading">${getIcon('zap')}<p>Loading content...</p></div>`;

    const content = await loadContent(item.path);
    if (!content) {
      this.body.innerHTML = `<div class="error" style="padding:40px;text-align:center;">${getIcon('alert')}<p style="margin-top:12px;color:var(--text-2)">Failed to load content</p><a href="${item.url}" target="_blank" rel="noopener" style="color:var(--accent-light);margin-top:16px;display:inline-block">View on GitHub</a></div>`;
      return;
    }

    this.currentContent = content;
    const isMD = item.name.endsWith('.md') || item.name.endsWith('.prompt.md');
    const isPy = item.name.endsWith('.py');
    const isYml = item.name.endsWith('.yaml') || item.name.endsWith('.yml');

    if (isMD) this.body.innerHTML = renderMarkdown(content);
    else if (isPy) this.body.innerHTML = renderCode(content, 'Python');
    else if (isYml) this.body.innerHTML = renderCode(content, 'YAML');
    else this.body.innerHTML = renderCode(content, 'Text');
  },

  openDir(item) {
    this.el.classList.add('open');
    document.body.style.overflow = 'hidden';
    this.currentPath = null;
    this.currentContent = null;

    this.title.textContent = item.name;
    this.subtitle.textContent = item.category || 'Directory';
    this.icon.innerHTML = getIcon(item.icon || 'folder');
    $('#modalGithubBtn').href = item.url;
    $('#modalDownloadBtn').href = item.url;

    // Show file listing
    const files = item.files || [];
    if (!files.length) {
      this.body.innerHTML = `<div class="md-viewer" style="padding:40px;text-align:center;">${getIcon('folder')}<p style="margin-top:12px;color:var(--text-3)">Directory listing not available</p><a href="${item.url}" target="_blank" rel="noopener" style="color:var(--accent-light);margin-top:16px;display:inline-block">View on GitHub</a></div>`;
      return;
    }

    this.body.innerHTML = `
      <div class="md-viewer" style="padding:24px 0;">
        <p style="color:var(--text-3);margin-bottom:16px;font-size:0.875rem;">${files.length} item${files.length>1?'s':''}</p>
        ${files.map(f => `
          <div class="item-card" style="margin-bottom:8px;" onclick="Modal.open({name:'${f.name.replace(/'/g,"\\'")}',category:'${item.name.replace(/'/g,"\\'")}',icon:'${f.icon||'file'}',path:'${f.path}',url:'${f.url}'${f.categorySlug?`,categorySlug:'${f.categorySlug}'`:''}})">
            <div class="item-icon">${getIcon(f.icon||'file')}</div>
            <div class="item-info">
              <div class="item-name">${f.name}</div>
              <div class="item-path">${f.path}</div>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  },

  close() {
    this.el.classList.remove('open');
    document.body.style.overflow = '';
    setTimeout(() => { this.body.innerHTML = ''; this.currentContent = null; }, 300);
  }
};

// ============================================
// Search
// ============================================
const Search = {
  overlay: null,
  input: null,
  results: null,

  init() {
    this.overlay = $('#searchOverlay');
    this.input = $('#searchInput');
    this.results = $('#searchResults');

    // Toggle with Cmd/Ctrl+K or click search icon
    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        this.toggle();
      }
      if (e.key === 'Escape' && this.overlay.classList.contains('open')) {
        this.close();
      }
    });

    // Global search input in navbar
    $('#globalSearchInput').addEventListener('focus', () => {
      this.open();
      setTimeout(() => this.input.focus(), 50);
    });

    this.overlay.addEventListener('click', (e) => {
      if (e.target === this.overlay) this.close();
    });

    this.input.addEventListener('input', () => this.perform());
  },

  open() { this.overlay.classList.add('open'); this.input.focus(); },
  close() { this.overlay.classList.remove('open'); this.input.value = ''; this.results.innerHTML = ''; },
  toggle() { this.overlay.classList.contains('open') ? this.close() : this.open(); },

  perform() {
    const q = this.input.value.trim().toLowerCase();
    if (!q) { this.results.innerHTML = ''; return; }

    const matches = allItems.filter(item =>
      item.name.toLowerCase().includes(q) ||
      (item.category && item.category.toLowerCase().includes(q)) ||
      (item.path && item.path.toLowerCase().includes(q))
    ).slice(0, 12);

    if (!matches.length) {
      this.results.innerHTML = '<div class="search-empty">No results found</div>';
      return;
    }

    this.results.innerHTML = matches.map((item, idx) => `
      <div class="search-result" data-search-idx="${idx}">
        <div class="search-result-icon">${getIcon(item.icon || 'file')}</div>
        <div class="search-result-info">
          <div class="search-result-name">${item.name}</div>
          <div class="search-result-path">${item.path || ''}</div>
        </div>
      </div>
    `).join('');

    // Add click handlers after rendering
    $$('.search-result', this.results).forEach((el, idx) => {
      el.onclick = () => {
        Modal.open(matches[idx]);
        this.close();
      };
    });
  }
};

// ============================================
// Theme
// ============================================
function initTheme() {
  const saved = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (saved === 'light' || (!saved && !prefersDark)) {
    document.documentElement.setAttribute('data-theme', 'light');
  }
  $('#themeToggle').onclick = () => {
    const isDark = !document.documentElement.hasAttribute('data-theme');
    if (isDark) document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', 'light');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  };
}

// ============================================
// Navbar Scroll
// ============================================
function initNavbar() {
  const nav = $('#navbar');
  let ticking = false;
  window.addEventListener('scroll', () => {
    if (!ticking) {
      requestAnimationFrame(() => {
        nav.classList.toggle('scrolled', window.scrollY > 20);
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });

  // Mobile menu
  $('#mobileToggle').onclick = () => $('#mobileMenu').classList.add('open');
  $('#mobileMenu').onclick = (e) => { if(e.target.id==='mobileMenu') $('#mobileMenu').classList.remove('open'); };
  $$('.mobile-menu-link').forEach(l => l.onclick = () => $('#mobileMenu').classList.remove('open'));

  // Active section highlighting
  const sections = $$('section[id]');
  const links = $$('.nav-link');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        links.forEach(l => l.classList.toggle('active', l.getAttribute('href')==='#'+entry.target.id));
      }
    });
  }, { rootMargin: '-50% 0px -50% 0px' });
  sections.forEach(s => observer.observe(s));
}

function scrollToTop() { window.scrollTo({ top:0, behavior:'smooth' }); }

// ============================================
// Lazy Loading
// ============================================
function initLazyLoad() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { rootMargin: '50px' });
  $$('.lazy').forEach(el => observer.observe(el));
}

// ============================================
// Render Functions
// ============================================
function renderSkills(items) {
  const grid = $('#skillsGrid');
  // Group by category
  const cats = {};
  items.forEach(item => {
    const c = item.category || 'Other';
    (cats[c] = cats[c] || []).push(item);
  });

  grid.innerHTML = Object.entries(cats).map(([cat, skills], i) => {
    const dirPath = 'skills/' + (skills[0].categorySlug || cat.toLowerCase().replace(/ /g,'-'));
    const dirUrl = ghUrl(dirPath, 'dir');
    // Build files array for directory modal
    const filesJs = skills.map(s => `{name:'${s.name.replace(/'/g,"\\'")}',path:'${s.path}',url:'${s.url}',icon:'${s.icon||'file'}'}`).join(',');
    return `
    <div class="card lazy" style="animation-delay:${i*50}ms" onclick="Modal.open({name:'${cat.replace(/'/g,"\\'")}',category:'Skills',icon:'${skills[0].icon||'folder'}',url:'${dirUrl}',isDir:true,files:[${filesJs}]})">
      <div class="card-icon">${getIcon(skills[0].icon||'folder')}</div>
      <div class="card-title">${cat}</div>
      <div class="card-desc">${skills.length} skill${skills.length>1?'s':''} for LLM engineering</div>
      <div class="card-meta">
        <span class="tag">${skills.length} items</span>
        <span class="tag">Skills</span>
      </div>
    </div>
  `}).join('');

  $('#skillsCount').textContent = `${items.length} skills`;
  $('#statSkills').textContent = items.length;
}

function renderScripts(items) {
  const grid = $('#scriptsGrid');
  grid.innerHTML = items.map((item, i) => `
    <div class="item-card lazy" style="animation-delay:${i*30}ms" onclick="Modal.open({name:'${item.name.replace(/'/g,"\\'")}',category:'${item.category||''}',icon:'${item.icon||'terminal'}',path:'${item.path}',url:'${item.url}'})">
      <div class="item-icon">${getIcon(item.icon||'terminal')}</div>
      <div class="item-info">
        <div class="item-name">${item.name}</div>
        <div class="item-path">${item.path}</div>
      </div>
      <div class="item-actions">
        <button class="action-btn" onclick="event.stopPropagation();Modal.open({name:'${item.name.replace(/'/g,"\\'")}',category:'${item.category||''}',icon:'${item.icon||'terminal'}',path:'${item.path}',url:'${item.url}'})">${getIcon('external')}</button>
      </div>
    </div>
  `).join('');
  $('#scriptsCount').textContent = `${items.length} scripts`;
  $('#statScripts').textContent = items.length;
}

function renderCode(items) {
  const grid = $('#codeGrid');
  grid.innerHTML = items.map((item, i) => `
    <div class="item-card lazy" style="animation-delay:${i*30}ms" onclick="Modal.open({name:'${item.name.replace(/'/g,"\\'")}',category:'${item.category||''}',icon:'${item.icon||'fileCode'}',path:'${item.path}',url:'${item.url}'})">
      <div class="item-icon">${getIcon(item.icon||'fileCode')}</div>
      <div class="item-info">
        <div class="item-name">${item.name}</div>
        <div class="item-path">${item.path}</div>
      </div>
      <div class="item-actions">
        <button class="action-btn" onclick="event.stopPropagation();Modal.open({name:'${item.name.replace(/'/g,"\\'")}',category:'${item.category||''}',icon:'${item.icon||'fileCode'}',path:'${item.path}',url:'${item.url}'})">${getIcon('external')}</button>
      </div>
    </div>
  `).join('');
  $('#codeCount').textContent = `${items.length} files`;
  $('#statCode').textContent = items.length;
}

function renderPipelines(items) {
  const grid = $('#pipelinesGrid');
  grid.innerHTML = items.map((item, i) => `
    <div class="card lazy" style="animation-delay:${i*40}ms" onclick="window.open('${item.url}','_blank')">
      <div class="card-icon">${getIcon(item.icon||'gitBranch')}</div>
      <div class="card-title">${item.name}</div>
      <div class="card-desc">Data, training, evaluation, and deployment workflows</div>
      <div class="card-meta">
        <span class="tag">Pipeline</span>
        ${item.hasConfigs ? '<span class="tag">Configs</span>' : ''}
      </div>
    </div>
  `).join('');
  $('#pipelinesCount').textContent = `${items.length} pipelines`;
  $('#statPipelines').textContent = items.length;
}

// ============================================
// Data Processing
// ============================================
function processSkills(raw) {
  const all = [];
  for (const catDir of raw) {
    if (catDir.type !== 'dir' || !catDir.children) continue;
    const cat = catDir.name;
    for (const child of catDir.children) {
      if (child.type === 'dir') {
        // Subdirectory - add all files inside
        for (const file of (child.children || [])) {
          if (file.type === 'file' && (file.name.endsWith('.prompt.md') || file.name.endsWith('.md'))) {
            const name = file.name.replace('.prompt.md','').replace('.md','');
            all.push({ name: fmtName(name), category: fmtName(cat), categorySlug: cat, path: file.path, url: ghUrl(file.path, 'file'), type: 'skill', icon: CAT_ICONS[cat] || 'folder' });
          }
        }
      } else if (child.type === 'file' && (child.name.endsWith('.prompt.md') || child.name.endsWith('.md'))) {
        const name = child.name.replace('.prompt.md','').replace('.md','');
        all.push({ name: fmtName(name), category: fmtName(cat), categorySlug: cat, path: child.path, url: ghUrl(child.path, 'file'), type: 'skill', icon: CAT_ICONS[cat] || 'folder' });
      }
    }
  }
  return all;
}

function processFiles(raw) {
  const all = [];
  for (const dir of raw) {
    if (dir.type !== 'dir' || !dir.children) continue;
    const cat = dir.name;
    for (const file of dir.children) {
      if (file.type === 'file') {
        all.push({ name: file.name, category: fmtName(cat), path: file.path, url: ghUrl(file.path, 'file'), type: 'file', icon: file.name.endsWith('.py')?'fileCode':(file.name.endsWith('.md')?'fileText':'file') });
      }
    }
  }
  return all;
}

function processPipelines(raw) {
  return raw.filter(d=>d.type==='dir').map(d=>({
    name: fmtName(d.name), path: d.path, url: ghUrl(d.path,'dir'),
    type: 'pipeline', category: 'Pipelines',
    hasConfigs: d.children?.some(c=>c.name==='configs'),
    icon: 'gitBranch'
  }));
}

// ============================================
// Main Init
// ============================================
async function init() {
  console.log('LLM-Whisperer v2 — Initializing...');

  initTheme();
  initNavbar();
  Modal.init();
  Search.init();

  try {
    // Load static data
    const res = await fetch('js/data.json');
    const data = await res.json();

    const skills = processSkills(data.skills || []);
    const scripts = processFiles(data.scripts || []);
    const code = processFiles(data.sample_code || []);
    const pipelines = processPipelines(data.pipelines || []);

    allItems = [...skills, ...scripts, ...code, ...pipelines];

    renderSkills(skills);
    renderScripts(scripts);
    renderCode(code);
    renderPipelines(pipelines);

    // Hide loader
    setTimeout(() => {
      $('#loader').classList.add('hidden');
      initLazyLoad();
    }, 500);

    console.log(`Loaded: ${skills.length} skills, ${scripts.length} scripts, ${code.length} code files, ${pipelines.length} pipelines`);
  } catch (e) {
    console.error('Failed to load data:', e);
    $('#loader .loader-text').textContent = 'Failed to load. Refresh to retry.';
  }
}

// Start
document.addEventListener('DOMContentLoaded', init);
