import React from 'react';
import './ConversationHistory.css';

function ConversationHistory({ messages }) {
  const userMessages = messages.filter(m => m.role === 'user');
  const recent = userMessages.slice(-3).reverse();

  return (
    <div className="conversation-history">
      <h3 className="history-title">This Session</h3>
      {userMessages.length > 0 ? (
        <>
          <div className="message-count">
            <strong>{userMessages.length}</strong> queries
          </div>
          {recent.length > 0 && (
            <div className="recent-queries">
              <h4 className="recent-title">Recent:</h4>
              <ul className="queries-list">
                {recent.map((msg, i) => (
                  <li key={i} className="query-item">
                    {msg.content.length > 60 ? msg.content.substring(0, 60) + '...' : msg.content}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      ) : (
        <div className="no-messages">Start a conversation to see history.</div>
      )}
    </div>
  );
}

export default ConversationHistory;
