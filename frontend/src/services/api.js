import axios from 'axios';

// Use environment variable if set, otherwise use relative path for Docker setup
const API_BASE_URL = process.env.REACT_APP_API_URL
  ? `${process.env.REACT_APP_API_URL}/api`
  : '/api';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 60000, // 60 second timeout to accommodate backend retries (increased from 5s)
    });
  }

  setSessionId(sessionId) {
    if (sessionId) {
      this.client.defaults.headers['X-Session-ID'] = sessionId;
    }
  }

  async healthCheck() {
    const response = await this.client.get('/health');
    return response.data;
  }

  async getStatus(sessionId) {
    this.setSessionId(sessionId);
    const response = await this.client.get('/status');
    return response.data;
  }

  async search(query, sessionId, options = {}) {
    this.setSessionId(sessionId);
    const response = await this.client.post('/search', {
      query,
      ...options,
    });
    return response.data;
  }

  async createNewSession() {
    const response = await this.client.post('/conversation/new');
    return response.data;
  }

  async getConversationHistory(sessionId, limit = 10) {
    this.setSessionId(sessionId);
    const response = await this.client.get('/conversation/history', {
      params: { limit },
    });
    return response.data;
  }

  async getSettings() {
    const response = await this.client.get('/settings');
    return response.data;
  }

  async indexDocuments(dataDir = '/data', forceReindex = false) {
    const response = await this.client.post('/index', {
      data_dir: dataDir,
      force_reindex: forceReindex,
    });
    return response.data;
  }

  async getIndexingStatus() {
    const response = await this.client.get('/index/status');
    return response.data;
  }

  async submitFeedback(feedbackData) {
    const response = await this.client.post('/feedback', feedbackData);
    return response.data;
  }
}

const api = new ApiService();
export default api;
