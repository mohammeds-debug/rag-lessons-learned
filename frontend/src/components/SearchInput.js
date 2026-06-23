import React, { useState, useEffect } from 'react';
import './SearchInput.css';

const placeholders = [
  'How does attention mechanism work in transformers?',
  'Explain the difference between BERT and GPT...',
  'What is RLHF and how is it used?',
  'How do diffusion models generate images?',
  'What are the key ideas behind RAG systems?',
  'Summarize recent advances in vision transformers...',
];

function SearchInput({ onSearch, disabled }) {
  const [query, setQuery] = useState('');
  const [placeholderIndex, setPlaceholderIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setPlaceholderIndex(i => (i + 1) % placeholders.length);
    }, 3500);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !disabled) {
      onSearch(query);
      setQuery('');
    }
  };

  return (
    <form className="search-input-container" onSubmit={handleSubmit}>
      <input
        type="text"
        className="search-input"
        placeholder={placeholders[placeholderIndex]}
        value={query}
        onChange={e => setQuery(e.target.value)}
        disabled={disabled}
      />
      <button
        type="submit"
        className="search-button"
        disabled={disabled || !query.trim()}
        aria-label="Search"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="8"></circle>
          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
      </button>
    </form>
  );
}

export default SearchInput;
