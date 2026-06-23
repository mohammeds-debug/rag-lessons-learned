import React from 'react';
import './RecommendedQuestions.css';

const questions = [
  'How does the attention mechanism work in Transformers?',
  'What is RLHF and how does it align language models?',
  'Explain how RAG improves language model accuracy',
  'What are the key innovations in diffusion models?',
  'How do Vision Transformers (ViT) compare to CNNs?',
];

function RecommendedQuestions({ onQuestionClick }) {
  return (
    <div className="recommended-questions">
      <p className="recommended-subtitle">Click a question to get started</p>
      <div className="questions-grid">
        {questions.map((q, i) => (
          <button key={i} className="question-card" onClick={() => onQuestionClick(q)}>
            <span className="question-icon">→</span>
            <span className="question-text">{q}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default RecommendedQuestions;
