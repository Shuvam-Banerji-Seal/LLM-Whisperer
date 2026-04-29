/* ============================================
   LLM-Whisperer - Static Data Loader (No API calls)
   ============================================ */

const REPO = 'Shuvam-Banerji-Seal/LLM-Whisperer';
const GH = 'https://github.com/' + REPO;

function ghUrl(path, type) {
  return type === 'dir' ? `${GH}/tree/main/${path}` : `${GH}/blob/main/${path}`;
}

function iconForCategory(cat) {
  const m = {
    rag:'database','rag-advanced':'database','agentic':'users','agents':'bot',
    inference:'zap','fine-tuning':'settings','production-ops':'settings',
    safety:'shield','advanced-architectures':'cpu','advanced-reasoning':'brain',
    foundational:'book','llm-engineering':'code-2','data-preprocessing':'filter',
    datasets:'database','evaluation':'clipboard-check','templates':'layout',
    workflows:'git-branch','fast-inference':'zap','training-optimization':'trending-up',
    'code-generation':'code','diffusion':'sparkles','huggingface':'heart',
    'image-generation':'image','infrastructure-deployment':'cloud',
    'knowledge-systems':'globe','long-context':'text-cursor-input',
    'model-merging':'git-merge','moe':'layers','multimodal':'image',
    'prompt-engineering':'message-square','quantization':'minimize-2',
    'security-governance':'shield','specialized-ml-techniques':'cpu',
    'time-series':'clock','transformers':'box','turboquant':'zap',
    'video-generation':'video', 'minimal':'code-2', 'end_to_end':'git-branch',
    'agent_patterns':'users', 'integration_examples':'globe', 'reference_apps':'layout',
  };
  return m[cat] || 'folder';
}

function iconForFile(name) {
  if (name.endsWith('.py')) return 'file-code';
  if (name.endsWith('.md')) return 'file-text';
  if (name.endsWith('.yaml')||name.endsWith('.yml')) return 'file';
  if (name.endsWith('.ipynb')) return 'notebook';
  if (name.endsWith('.json')) return 'file-json';
  return 'file';
}

function fmtName(s) {
  return s.split(/[-_]/).map(w=>w.charAt(0).toUpperCase()+w.slice(1)).join(' ');
}

// Process raw file tree into display items
function processSkills(raw) {
  const all = [];
  for (const catDir of raw) {
    if (catDir.type !== 'dir' || !catDir.children) continue;
    const cat = catDir.name;
    for (const skill of catDir.children) {
      if (skill.type !== 'dir' && (skill.name.endsWith('.prompt.md') || skill.name.endsWith('.md'))) {
        const name = skill.name.replace('.prompt.md','').replace('.md','');
        all.push({
          name: fmtName(name),
          category: fmtName(cat),
          categorySlug: cat,
          path: skill.path,
          url: ghUrl(skill.path, 'file'),
          type: 'skill',
          icon: iconForCategory(cat),
        });
      }
    }
  }
  return all;
}

function processSampleCode(raw) {
  const all = [];
  for (const dir of raw) {
    if (dir.type !== 'dir' || !dir.children) continue;
    const cat = dir.name;
    for (const file of dir.children) {
      if (file.type === 'file') {
        all.push({
          name: file.name,
          category: fmtName(cat),
          path: file.path,
          url: ghUrl(file.path, 'file'),
          type: 'sample-code',
          icon: iconForFile(file.name),
        });
      }
    }
  }
  return all;
}

function processScripts(raw) {
  const all = [];
  for (const dir of raw) {
    if (dir.type !== 'dir' || !dir.children) continue;
    const cat = dir.name;
    for (const file of dir.children) {
      if (file.type === 'file' && file.name.endsWith('.py')) {
        all.push({
          name: file.name.replace('.py','').replace(/_/g,' '),
          category: fmtName(cat),
          path: file.path,
          url: ghUrl(file.path, 'file'),
          type: 'script',
          icon: 'terminal',
        });
      }
    }
  }
  return all;
}

function processPipelines(raw) {
  return raw.map(dir => ({
    name: fmtName(dir.name),
    path: dir.path,
    url: ghUrl(dir.path, 'dir'),
    type: 'pipeline',
    category: 'Pipelines',
    hasConfigs: dir.children ? dir.children.some(c => c.name === 'configs') : false,
    icon: 'git-branch',
  }));
}

function processNotebooks(raw) {
  return raw.filter(d => d.type === 'dir').map(d => ({
    name: d.name,
    path: d.path,
    url: ghUrl(d.path, 'dir'),
    type: 'notebook',
    icon: 'notebook',
    category: 'Notebooks',
  }));
}

function processConfigs(raw) {
  return raw.filter(d => d.type === 'dir').map(d => ({
    name: d.name,
    path: d.path,
    url: ghUrl(d.path, 'dir'),
    type: 'config',
    icon: 'settings',
    category: 'Configs',
  }));
}