/* ============================================
   LLM-Whisperer - GitHub API Module
   ============================================ */

const API = {
  base: 'https://api.github.com',
  repo: 'Shuvam-Banerji-Seal/LLM-Whisperer',
  cache: new Map(),

  // Rate limiting
  rateLimit: {
    remaining: 60,
    reset: 0,
  },

  /**
   * Make a request to GitHub API
   */
  async request(endpoint, options = {}) {
    const url = `${this.base}${endpoint}`;

    // Check cache
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

      // Update rate limit info
      this.rateLimit.remaining = parseInt(
        response.headers.get('X-RateLimit-Remaining') || '60'
      );
      this.rateLimit.reset = parseInt(
        response.headers.get('X-RateLimit-Reset') || '0'
      );

      if (!response.ok) {
        if (response.status === 403 && this.rateLimit.remaining === 0) {
          const resetTime = new Date(this.rateLimit.reset * 1000);
          throw new Error(
            `Rate limit exceeded. Resets at ${resetTime.toLocaleTimeString()}`
          );
        }
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      // Cache the result (expire after 5 minutes)
      this.cache.set(url, data);
      setTimeout(() => this.cache.delete(url), 5 * 60 * 1000);

      return data;
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  },

  /**
   * Get contents of a directory
   */
  async getContents(path = '') {
    const endpoint = `/repos/${this.repo}/contents/${path}`;
    return this.request(endpoint);
  },

  /**
   * Get file content (decoded from base64)
   */
  async getFileContent(path) {
    const data = await this.getContents(path);
    if (data.content) {
      return atob(data.content.replace(/\n/g, ''));
    }
    return null;
  },

  /**
   * Get repository info
   */
  async getRepoInfo() {
    return this.request(`/repos/${this.repo}`);
  },

  /**
   * Get all skills from skills/ directory
   */
  async getSkills() {
    try {
      const skillsDirs = await this.getContents('skills');

      const skills = [];
      for (const dir of skillsDirs) {
        if (dir.type === 'dir') {
          try {
            const files = await this.getContents(dir.path);
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

      return skills;
    } catch (error) {
      console.error('Failed to load skills:', error);
      return this.getFallbackSkills();
    }
  },

  /**
   * Get sample code files
   */
  async getSampleCode() {
    try {
      const result = [];
      const dirs = ['minimal', 'end_to_end', 'reference_apps', 'agent_patterns', 'integration_examples'];

      for (const dir of dirs) {
        try {
          const files = await this.getContents(`sample_code/${dir}`);
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

      return result;
    } catch (error) {
      console.error('Failed to load sample code:', error);
      return this.getFallbackSampleCode();
    }
  },

  /**
   * Get scripts
   */
  async getScripts() {
    try {
      const result = [];
      const dirs = ['data', 'model', 'eval', 'deploy'];

      for (const dir of dirs) {
        try {
          const files = await this.getContents(`scripts/${dir}`);
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

      // Also check root scripts
      try {
        const files = await this.getContents('scripts');
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
      } catch (e) {
        // Ignore
      }

      return result;
    } catch (error) {
      console.error('Failed to load scripts:', error);
      return [];
    }
  },

  /**
   * Get pipelines
   */
  async getPipelines() {
    try {
      const result = [];
      const dirs = ['data', 'training', 'evaluation', 'deployment'];

      for (const dir of dirs) {
        try {
          const contents = await this.getContents(`pipelines/${dir}`);
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

      return result;
    } catch (error) {
      console.error('Failed to load pipelines:', error);
      return [];
    }
  },

  /**
   * Helper: Format name (kebab-case to Title Case)
   */
  formatName(str) {
    return str
      .split(/[-_]/)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  },

  /**
   * Helper: Get icon for category
   */
  getIconForCategory(category) {
    const icons = {
      'rag': 'database',
      'agentic': 'users',
      'inference': 'zap',
      'fine-tuning': 'tuning',
      'production-ops': 'settings',
      'safety': 'shield',
      'advanced-architectures': 'building',
      'advanced-reasoning': 'brain',
      'foundational': 'book',
      'llm-engineering': 'code-2',
      'data-preprocessing': 'filter',
      'datasets': 'database',
      'evaluation': 'bar-chart-2',
      'templates': 'layout',
      'workflows': 'git-branch',
    };
    return icons[category] || 'file-text';
  },

  /**
   * Helper: Get icon for file type
   */
  getIconForFile(filename) {
    if (filename.endsWith('.py')) return 'file-code';
    if (filename.endsWith('.md')) return 'file-text';
    if (filename.endsWith('.yaml') || filename.endsWith('.yml')) return 'file-settings';
    if (filename.endsWith('.ipynb')) return 'notebook';
    return 'file';
  },

  /**
   * Fallback data if API fails
   */
  getFallbackSkills() {
    const categories = [
      'rag', 'agentic', 'inference', 'fine-tuning', 'production-ops',
      'safety', 'advanced-architectures', 'advanced-reasoning'
    ];

    const skills = [];
    categories.forEach((cat) => {
      skills.push({
        name: this.formatName(cat),
        category: this.formatName(cat),
        categorySlug: cat,
        type: 'skill',
        icon: this.getIconForCategory(cat),
        url: `https://github.com/${this.repo}/tree/main/skills/${cat}`,
      });
    });

    return skills;
  },

  getFallbackSampleCode() {
    return [
      { name: 'Minimal RAG', type: 'sample-code', category: 'Minimal', icon: 'database' },
      { name: 'Minimal Agent', type: 'sample-code', category: 'Minimal', icon: 'users' },
      { name: 'E2E RAG Pipeline', type: 'sample-code', category: 'End to End', icon: 'git-branch' },
      { name: 'Tool Patterns', type: 'sample-code', category: 'Agent Patterns', icon: 'tool' },
    ];
  }
};
