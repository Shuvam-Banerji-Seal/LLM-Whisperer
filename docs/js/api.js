/* ============================================
   LLM-Whisperer - GitHub API Module
   ============================================ */

const API = {
  base: 'https://api.github.com',
  repo: 'Shuvam-Banerji-Seal/LLM-Whisperer',
  cache: new Map(),
  rateLimit: { remaining: 60, reset: 0 },

  async request(endpoint, options = {}) {
    const url = `${this.base}${endpoint}`;

    if (this.cache.has(url)) {
      return this.cache.get(url);
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          Accept: 'application/vnd.github.v3+json',
          'User-Agent': 'LLM-Whisperer-Website',
          ...options.headers,
        },
      });

      this.rateLimit.remaining = parseInt(
        response.headers.get('X-RateLimit-Remaining') || '60'
      );
      this.rateLimit.reset = parseInt(
        response.headers.get('X-RateLimit-Reset') || '0'
      );

      if (!response.ok) {
        if (response.status === 403 && this.rateLimit.remaining === 0) {
          const resetTime = new Date(this.rateLimit.reset * 1000);
          throw new Error(`Rate limit exceeded. Resets at ${resetTime.toLocaleTimeString()}`);
        }
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      this.cache.set(url, data);
      setTimeout(() => this.cache.delete(url), 5 * 60 * 1000);
      return data;
    } catch (error) {
      if (error.name === 'NetworkError' || error.message.includes('NetworkError')) {
        console.warn('Network error, using fallback data');
        return null;
      }
      console.error('API Request failed:', error);
      throw error;
    }
  },

  async getContents(path = '') {
    const endpoint = `/repos/${this.repo}/contents/${path}`;
    return this.request(endpoint);
  },

  async getFileContent(path) {
    const data = await this.getContents(path);
    if (data && data.content) {
      return atob(data.content.replace(/\n/g, ''));
    }
    return null;
  },

  async getRepoInfo() {
    return this.request(`/repos/${this.repo}`);
  },

  async getSkills() {
    try {
      const skillsDirs = await this.getContents('skills');
      if (!skillsDirs) return this.getFallbackSkills();

      const skills = [];
      for (const dir of skillsDirs) {
        if (dir.type === 'dir') {
          try {
            const files = await this.getContents(dir.path);
            if (!files) continue;

            const promptFiles = files.filter(
              (f) => f.name.endsWith('.prompt.md') || f.name.endsWith('.md')
            );

            for (const file of promptFiles) {
              const category = dir.name;
              const name = file.name.replace('.prompt.md', '').replace(/-/g, ' ');

              skills.push({
                name: this.formatName(name),
                category: this.formatName(category),
                categorySlug: category,
                path: file.path,
                url: file.html_url,
                type: 'skill',
                icon: this.getIconForCategory(category),
              });
            }
          } catch (e) {
            console.warn(`Failed to load skills from ${dir.path}:`, e);
          }
        }
      }

      return skills.length > 0 ? skills : this.getFallbackSkills();
    } catch (error) {
      console.warn('Failed to load skills, using fallback:', error);
      return this.getFallbackSkills();
    }
  },

  async getSampleCode() {
    try {
      const result = [];
      const dirs = ['minimal', 'end_to_end', 'reference_apps', 'agent_patterns', 'integration_examples'];

      for (const dir of dirs) {
        try {
          const files = await this.getContents(`sample_code/${dir}`);
          if (!files) continue;

          for (const file of files) {
            if (file.type === 'file') {
              result.push({
                name: file.name,
                path: file.path,
                url: file.html_url,
                type: 'sample-code',
                category: this.formatName(dir),
                icon: this.getIconForFile(file.name),
              });
            }
          }
        } catch (e) {
          console.warn(`Failed to load sample code from ${dir}:`, e);
        }
      }

      return result.length > 0 ? result : this.getFallbackSampleCode();
    } catch (error) {
      console.warn('Failed to load sample code, using fallback:', error);
      return this.getFallbackSampleCode();
    }
  },

  async getScripts() {
    try {
      const result = [];
      const dirs = ['data', 'model', 'eval', 'deploy'];

      for (const dir of dirs) {
        try {
          const files = await this.getContents(`scripts/${dir}`);
          if (!files) continue;

          for (const file of files) {
            if (file.type === 'file' && file.name.endsWith('.py')) {
              result.push({
                name: file.name.replace('.py', '').replace(/_/g, ' '),
                path: file.path,
                url: file.html_url,
                type: 'script',
                category: this.formatName(dir),
                icon: 'terminal',
              });
            }
          }
        } catch (e) {
          // Directory might not exist
        }
      }

      try {
        const files = await this.getContents('scripts');
        if (files) {
          for (const file of files) {
            if (file.type === 'file' && file.name.endsWith('.py')) {
              result.push({
                name: file.name.replace('.py', '').replace(/_/g, ' '),
                path: file.path,
                url: file.html_url,
                type: 'script',
                category: 'General',
                icon: 'terminal',
              });
            }
          }
        }
      } catch (e) {
        // Ignore
      }

      return result.length > 0 ? result : this.getFallbackScripts();
    } catch (error) {
      console.warn('Failed to load scripts, using fallback:', error);
      return this.getFallbackScripts();
    }
  },

  async getPipelines() {
    try {
      const result = [];
      const dirs = ['data', 'training', 'evaluation', 'deployment'];

      for (const dir of dirs) {
        try {
          const contents = await this.getContents(`pipelines/${dir}`);
          if (!contents) continue;

          const hasConfigs = contents.some(
            (f) => f.type === 'dir' && f.name === 'configs'
          );

          result.push({
            name: this.formatName(dir),
            path: `pipelines/${dir}`,
            url: `https://github.com/${this.repo}/tree/main/pipelines/${dir}`,
            type: 'pipeline',
            category: 'Pipelines',
            hasConfigs,
            icon: 'git-branch',
          });
        } catch (e) {
          console.warn(`Failed to load pipeline ${dir}:`, e);
        }
      }

      return result.length > 0 ? result : this.getFallbackPipelines();
    } catch (error) {
      console.warn('Failed to load pipelines, using fallback:', error);
      return this.getFallbackPipelines();
    }
  },

  formatName(str) {
    return str
      .split(/[-_]/)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  },

  getIconForCategory(category) {
    const icons = {
      'rag': 'database',
      'rag-advanced': 'database',
      'agentic': 'users',
      'inference': 'zap',
      'fine-tuning': 'settings',
      'production-ops': 'settings',
      'safety': 'shield',
      'advanced-architectures': 'cpu',
      'advanced-reasoning': 'brain',
      'foundational': 'book',
      'llm-engineering': 'code-2',
      'data-preprocessing': 'filter',
      'datasets': 'database',
      'evaluation': 'bar-chart-2',
      'templates': 'layout',
      'workflows': 'git-branch',
      'agents': 'bot',
      'code-generation': 'code',
      'diffusion': 'sparkles',
      'evaluation': 'clipboard-check',
      'fast-inference': 'zap',
      'huggingface': 'heart',
      'image-generation': 'image',
      'infrastructure-deployment': 'cloud',
      'knowledge-systems': 'globe',
      'long-context': 'text-cursor-input',
      'model-merging': 'git-merge',
      'moe': 'layers',
      'multimodal': 'image',
      'prompt-engineering': 'message-square',
      'quantization': 'minimize-2',
      'security-governance': 'shield',
      'specialized-ml-techniques': 'cpu',
      'time-series': 'clock',
      'training-optimization': 'trending-up',
      'transformers': 'box',
      'turboquant': 'zap',
      'video-generation': 'video',
    };
    return icons[category] || 'folder';
  },

  getIconForFile(filename) {
    if (filename.endsWith('.py')) return 'file-code';
    if (filename.endsWith('.md')) return 'file-text';
    if (filename.endsWith('.yaml') || filename.endsWith('.yml')) return 'file';
    if (filename.endsWith('.ipynb')) return 'notebook';
    if (filename.endsWith('.json')) return 'file-json';
    return 'file';
  },

  getFallbackSkills() {
    return [
      { name: 'RAG Skills', category: 'RAG', categorySlug: 'rag', type: 'skill', icon: 'database', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/rag' },
      { name: 'Agentic Skills', category: 'Agentic', categorySlug: 'agentic', type: 'skill', icon: 'users', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/agentic' },
      { name: 'Inference Skills', category: 'Inference', categorySlug: 'inference', type: 'skill', icon: 'zap', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/inference' },
      { name: 'Fine-tuning Skills', category: 'Fine-tuning', categorySlug: 'fine-tuning', type: 'skill', icon: 'settings', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/fine-tuning' },
      { name: 'Production Ops', category: 'Production', categorySlug: 'production-ops', type: 'skill', icon: 'settings', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/production-ops' },
      { name: 'Safety Skills', category: 'Safety', categorySlug: 'safety', type: 'skill', icon: 'shield', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/safety' },
      { name: 'Advanced Architectures', category: 'Architectures', categorySlug: 'advanced-architectures', type: 'skill', icon: 'cpu', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/advanced-architectures' },
      { name: 'Advanced Reasoning', category: 'Reasoning', categorySlug: 'advanced-reasoning', type: 'skill', icon: 'brain', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/advanced-reasoning' },
      { name: 'Fast Inference', category: 'Fast Inference', categorySlug: 'fast-inference', type: 'skill', icon: 'zap', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/fast-inference' },
      { name: 'Training Optimization', category: 'Training', categorySlug: 'training-optimization', type: 'skill', icon: 'trending-up', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/training-optimization' },
      { name: 'Foundational Skills', category: 'Foundational', categorySlug: 'foundational', type: 'skill', icon: 'book', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/foundational' },
      { name: 'Evaluation Skills', category: 'Evaluation', categorySlug: 'evaluation', type: 'skill', icon: 'clipboard-check', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/skills/evaluation' },
    ];
  },

  getFallbackSampleCode() {
    return [
      { name: 'minimal_rag.py', category: 'Minimal', type: 'sample-code', icon: 'file-code', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/sample_code/minimal' },
      { name: 'minimal_agent.py', category: 'Minimal', type: 'sample-code', icon: 'file-code', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/sample_code/minimal' },
      { name: 'minimal_finetune.py', category: 'Minimal', type: 'sample-code', icon: 'file-code', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/sample_code/minimal' },
      { name: 'e2e_rag_pipeline.py', category: 'End to End', type: 'sample-code', icon: 'file-code', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/sample_code/end_to_end' },
      { name: 'e2e_finetune_pipeline.py', category: 'End to End', type: 'sample-code', icon: 'file-code', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/sample_code/end_to_end' },
      { name: 'e2e_deployment.py', category: 'End to End', type: 'sample-code', icon: 'file-code', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/sample_code/end_to_end' },
      { name: 'tool_patterns.py', category: 'Agent Patterns', type: 'sample-code', icon: 'file-code', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/sample_code/agent_patterns' },
      { name: 'planning_patterns.py', category: 'Agent Patterns', type: 'sample-code', icon: 'file-code', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/sample_code/agent_patterns' },
    ];
  },

  getFallbackScripts() {
    return [
      { name: 'download_dataset.py', category: 'Data', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts/data' },
      { name: 'validate_data.py', category: 'Data', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts/data' },
      { name: 'augment_data.py', category: 'Data', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts/data' },
      { name: 'export_model.py', category: 'Model', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts/model' },
      { name: 'quantize_model.py', category: 'Model', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts/model' },
      { name: 'merge_adapters.py', category: 'Model', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts/model' },
      { name: 'run_benchmark.py', category: 'Eval', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts/eval' },
      { name: 'judge_evaluation.py', category: 'Eval', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts/eval' },
      { name: 'build_container.py', category: 'Deploy', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts/deploy' },
      { name: 'health_check.py', category: 'Deploy', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts/deploy' },
      { name: 'setup_environment.py', category: 'General', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts' },
      { name: 'benchmark_inference.py', category: 'General', type: 'script', icon: 'terminal', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/scripts' },
    ];
  },

  getFallbackPipelines() {
    return [
      { name: 'Data Pipeline', category: 'Pipelines', type: 'pipeline', icon: 'git-branch', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/pipelines/data', hasConfigs: true },
      { name: 'Training Pipeline', category: 'Pipelines', type: 'pipeline', icon: 'git-branch', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/pipelines/training', hasConfigs: true },
      { name: 'Evaluation Pipeline', category: 'Pipelines', type: 'pipeline', icon: 'git-branch', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/pipelines/evaluation', hasConfigs: true },
      { name: 'Deployment Pipeline', category: 'Pipelines', type: 'pipeline', icon: 'git-branch', url: 'https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer/tree/main/pipelines/deployment', hasConfigs: true },
    ];
  }
};