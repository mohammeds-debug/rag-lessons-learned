import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import useTypingEffect from '../hooks/useTypingEffect';
import api from '../services/api';
import './SearchResults.css';

function SearchResults({ results, enableTyping = true, sessionId, messageIndex, query }) {
  const { displayedText, isTyping } = useTypingEffect(
    enableTyping ? results?.ai_response || '' : '',
    12
  );

  const [feedbackGiven, setFeedbackGiven] = useState(null);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);

  if (!results) return null;

  const textToDisplay = enableTyping ? displayedText : results.ai_response;
  const references = results.structured_references || [];
  const researchArea = results.research_area || null;

  const handleFeedback = async (rating) => {
    if (feedbackSubmitting || feedbackGiven) return;
    setFeedbackSubmitting(true);
    try {
      await api.submitFeedback({
        run_id: results.run_id || null,
        session_id: sessionId,
        message_index: messageIndex,
        rating,
        query,
      });
      setFeedbackGiven(rating);
    } catch {
      setFeedbackGiven(rating);
    } finally {
      setFeedbackSubmitting(false);
    }
  };

  return (
    <div className="search-results">
      {(textToDisplay || results.ai_response) && (
        <div className="ai-response-section">
          <div className="section-header">
            <h3 className="section-title">Answer based on Retrieved Papers</h3>
            {researchArea && <span className="topic-badge">{researchArea}</span>}
          </div>

          <div className="ai-response">
            <ReactMarkdown>{textToDisplay}</ReactMarkdown>
            {isTyping && <span className="typing-cursor">▊</span>}
          </div>

          {!isTyping && references.length > 0 && (
            <div className="references-section">
              <h4 className="references-title">Sources</h4>
              <div className="references-list">
                {references.map((ref, idx) => (
                  <div key={idx} className="reference-card">
                    <div className="reference-primary">
                      {ref.paper_title && (
                        <span className="ref-paper">{ref.paper_title}</span>
                      )}
                    </div>
                    <div className="reference-secondary">
                      {ref.section && (
                        <span className="ref-section">{ref.section}</span>
                      )}
                      {ref.page_number && (
                        <span className="ref-page">p. {ref.page_number}</span>
                      )}
                      {ref.source_file && (
                        <span className="ref-file">{ref.source_file}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!isTyping && (
            <div className="feedback-buttons">
              <button
                className={`feedback-btn ${feedbackGiven === 'positive' ? 'active positive' : ''}`}
                onClick={() => handleFeedback('positive')}
                disabled={feedbackSubmitting || feedbackGiven !== null}
                title="Helpful"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
                  fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
                </svg>
              </button>
              <button
                className={`feedback-btn ${feedbackGiven === 'negative' ? 'active negative' : ''}`}
                onClick={() => handleFeedback('negative')}
                disabled={feedbackSubmitting || feedbackGiven !== null}
                title="Not helpful"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
                  fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>
                </svg>
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default SearchResults;
