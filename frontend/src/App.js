import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import SearchInput from './components/SearchInput';
import SearchResults from './components/SearchResults';
import RecommendedQuestions from './components/RecommendedQuestions';
import ConversationHistory from './components/ConversationHistory';
import api from './services/api';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [status, setStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendUnavailable, setBackendUnavailable] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    const initialize = async () => {
      try {
        const newSessionResponse = await api.createNewSession();
        const newSessionId = newSessionResponse.session_id;
        setSessionId(newSessionId);
        const statusResponse = await api.getStatus(newSessionId);
        setStatus(statusResponse);
        setBackendUnavailable(false);
      } catch (err) {
        console.error('Failed to initialize:', err);
        setBackendUnavailable(true);
        setError('Failed to connect to backend. Please try again later.');
      }
    };
    initialize();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSearch = async (query) => {
    if (!query.trim() || !sessionId) return;

    setLoading(true);
    setError(null);
    setMessages(prev => [...prev, { role: 'user', content: query }]);

    try {
      const results = await api.search(query, sessionId, {
        use_enhanced_features: true,
        use_query_improvement: true,
        use_context_parsing: true,
      });
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: results.ai_response || '',
        results,
        query,
      }]);
    } catch (err) {
      console.error('Search failed:', err);
      setError('Search failed. Please try again.');
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Search failed. Please try again.',
        error: true,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleNewSession = async () => {
    try {
      const response = await api.createNewSession();
      setSessionId(response.session_id);
      setMessages([]);
      setError(null);
      const statusResponse = await api.getStatus(response.session_id);
      setStatus(statusResponse);
    } catch (err) {
      setError('Failed to create new session.');
    }
  };

  const isIndexed = backendUnavailable ? false : (status?.indexed !== false);

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <div className="header-left">
            <button
              className="hamburger-btn"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle conversation history"
            >
              <span className="hamburger-icon">☰</span>
            </button>
            <div className="app-branding">
              <h1 className="app-title">ResearchRAG</h1>
              <p className="app-subtitle">Semantic search over AI/ML papers</p>
            </div>
          </div>
          <div className="header-actions">
            {messages.length > 0 && (
              <button className="new-session-btn" onClick={handleNewSession}>
                New Session
              </button>
            )}
          </div>
        </header>

        <div className="main-content">
          <div className={`content-wrapper ${messages.length > 0 ? 'has-messages' : ''}`}>
            {!isIndexed ? (
              <div className="welcome-section">
                <h2 className="welcome-greeting">Welcome to ResearchRAG</h2>
                <h3 className="welcome-question">What would you like to explore?</h3>
                <div className="warning-message">
                  No papers indexed yet. Run <code>python backend/index_documents.py</code> to get started.
                </div>
              </div>
            ) : (
              <>
                {messages.length === 0 && (
                  <div className="welcome-section">
                    <h2 className="welcome-greeting">Welcome to ResearchRAG</h2>
                    <h3 className="welcome-question">What would you like to explore?</h3>
                    <RecommendedQuestions onQuestionClick={handleSearch} />
                  </div>
                )}

                {messages.length > 0 && (
                  <div className="messages-container">
                    {messages.map((message, index) => {
                      const isLatest = index === messages.length - 1;
                      return (
                        <div key={index} className={`message ${message.role}`}>
                          {message.role === 'user' ? (
                            <div className="user-message">
                              <div className="message-content">{message.content}</div>
                            </div>
                          ) : (
                            <div className="assistant-message">
                              {message.error ? (
                                <div className="error-message">{message.content}</div>
                              ) : (
                                <SearchResults
                                  results={message.results}
                                  enableTyping={isLatest && !loading}
                                  sessionId={sessionId}
                                  messageIndex={index}
                                  query={message.query}
                                />
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                    {loading && (
                      <div className="message assistant">
                        <div className="loading-indicator">
                          <div className="spinner"></div>
                          <span>Searching papers...</span>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </>
            )}
          </div>

          {isIndexed && (
            <div className="search-input-fixed">
              <SearchInput onSearch={handleSearch} disabled={loading || !isIndexed} />
            </div>
          )}

          {sidebarOpen && (
            <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
          )}

          <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
            <button
              className="sidebar-close-btn"
              onClick={() => setSidebarOpen(false)}
              aria-label="Close sidebar"
            >
              ×
            </button>
            <ConversationHistory messages={messages} />
          </aside>
        </div>

        {messages.length === 0 && (
          <footer className="footer">
            <p className="footer-disclaimer">
              Answers are generated from retrieved paper excerpts. Always verify claims
              against the original papers.
            </p>
          </footer>
        )}

        {error && (
          <div className="error-toast">
            {error}
            <button onClick={() => setError(null)}>×</button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
