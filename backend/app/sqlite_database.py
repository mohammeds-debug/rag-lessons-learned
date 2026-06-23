"""
SQLite database module for storing conversation sessions and metadata.
This is a lightweight alternative to PostgreSQL that works without additional setup.
"""

import os
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConversationDB:
    """SQLite-based database manager for conversation sessions and metadata."""

    def __init__(self, db_path: str = None):
        """Initialize SQLite database connection."""
        if db_path is None:
            # Use absolute path in project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(project_root, "conversations.db")

        self.db_path = db_path

        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # Create conversations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                
                # Create context_cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS context_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        query_hash TEXT NOT NULL,
                        original_query TEXT NOT NULL,
                        improved_query TEXT,
                        retrieved_context TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversations_session_id 
                    ON conversations(session_id, created_at)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_context_cache_session_id 
                    ON context_cache(session_id, created_at)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_context_cache_query_hash 
                    ON context_cache(query_hash)
                """)
                
                conn.commit()
                
        except Exception as e:
            print(f"Error initializing SQLite database: {e}")
            raise
    
    def create_session(self, session_id: str, metadata: Optional[Dict] = None) -> bool:
        """Create a new conversation session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions (session_id, metadata, last_activity)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (session_id, json.dumps(metadata) if metadata else None))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error creating session: {e}")
            return False
    
    def add_conversation_message(self, session_id: str, message_type: str, 
                               content: str, metadata: Optional[Dict] = None) -> bool:
        """Add a message to the conversation history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO conversations (session_id, message_type, content, metadata)
                    VALUES (?, ?, ?, ?)
                """, (session_id, message_type, content, json.dumps(metadata) if metadata else None))
                
                # Update session last activity
                cursor.execute("""
                    UPDATE sessions 
                    SET last_activity = CURRENT_TIMESTAMP 
                    WHERE session_id = ?
                """, (session_id,))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error adding conversation message: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT message_type, content, metadata, created_at
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (session_id, limit))
                
                messages = []
                for row in cursor.fetchall():
                    messages.append({
                        'role': row['message_type'],
                        'content': row['content'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                        'created_at': row['created_at']
                    })
                
                return list(reversed(messages))  # Return in chronological order
                
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return []
    
    def cache_context(self, session_id: str, query_hash: str, original_query: str,
                     improved_query: str, retrieved_context: str) -> bool:
        """Cache context for future reference."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO context_cache
                    (session_id, query_hash, original_query, improved_query,
                     retrieved_context, created_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (session_id, query_hash, original_query, improved_query,
                      retrieved_context))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error caching context: {e}")
            return False
    
    def get_cached_context(self, session_id: str, query_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached context for a query."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT original_query, improved_query, retrieved_context, created_at
                    FROM context_cache
                    WHERE session_id = ? AND query_hash = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (session_id, query_hash))

                row = cursor.fetchone()
                if row:
                    return {
                        'original_query': row['original_query'],
                        'improved_query': row['improved_query'],
                        'retrieved_context': row['retrieved_context'],
                        'created_at': row['created_at']
                    }
                return None

        except Exception as e:
            print(f"Error getting cached context: {e}")
            return None
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get conversation count
                cursor.execute("""
                    SELECT COUNT(*) as message_count
                    FROM conversations
                    WHERE session_id = ?
                """, (session_id,))
                message_count = cursor.fetchone()['message_count']
                
                # Get cached context count
                cursor.execute("""
                    SELECT COUNT(*) as cache_count
                    FROM context_cache
                    WHERE session_id = ?
                """, (session_id,))
                cache_count = cursor.fetchone()['cache_count']
                
                # Get session info
                cursor.execute("""
                    SELECT created_at, last_activity, metadata
                    FROM sessions
                    WHERE session_id = ?
                """, (session_id,))
                session_info = cursor.fetchone()
                
                return {
                    'session_id': session_id,
                    'message_count': message_count,
                    'cache_count': cache_count,
                    'created_at': session_info['created_at'] if session_info else None,
                    'last_activity': session_info['last_activity'] if session_info else None,
                    'metadata': json.loads(session_info['metadata']) if session_info and session_info['metadata'] else {}
                }
                
        except Exception as e:
            print(f"Error getting session stats: {e}")
            return {}
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all conversation sessions with basic info."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.session_id, s.created_at, s.last_activity, s.metadata,
                           COUNT(c.id) as message_count
                    FROM sessions s
                    LEFT JOIN conversations c ON s.session_id = c.session_id
                    GROUP BY s.session_id
                    ORDER BY s.last_activity DESC
                """)
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        'session_id': row['session_id'],
                        'created_at': row['created_at'],
                        'last_activity': row['last_activity'],
                        'message_count': row['message_count'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                    })
                
                return sessions
                
        except Exception as e:
            print(f"Error getting all sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its associated data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete associated data first
                cursor.execute("DELETE FROM context_cache WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def get_conversation_context_for_enhancement(self, session_id: str, limit: int = 5) -> str:
        """Get recent conversation context for query enhancement."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT message_type, content
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (session_id, limit * 2))  # Get more to filter properly
                
                messages = cursor.fetchall()
                context_parts = []
                
                # Build context from recent messages
                for message_type, content in reversed(messages):
                    if message_type == 'user':
                        context_parts.append(f"User: {content}")
                    elif message_type == 'assistant':
                        context_parts.append(f"Assistant: {content}")
                
                return "\n".join(context_parts[-limit:]) if context_parts else ""
                
        except Exception as e:
            print(f"Error getting conversation context: {e}")
            return ""
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old sessions and their associated data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get sessions to delete
                cursor.execute("""
                    SELECT session_id FROM sessions
                    WHERE last_activity < datetime('now', '-{} days')
                """.format(days_old))
                old_sessions = [row[0] for row in cursor.fetchall()]
                
                if not old_sessions:
                    return 0
                
                # Delete associated data
                placeholders = ','.join('?' * len(old_sessions))
                cursor.execute(f"""
                    DELETE FROM context_cache
                    WHERE session_id IN ({placeholders})
                """, old_sessions)
                
                cursor.execute(f"""
                    DELETE FROM conversations
                    WHERE session_id IN ({placeholders})
                """, old_sessions)
                
                cursor.execute(f"""
                    DELETE FROM sessions
                    WHERE session_id IN ({placeholders})
                """, old_sessions)
                
                conn.commit()
                return len(old_sessions)
                
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")
            return 0


def get_conversation_db() -> ConversationDB:
    """Get a configured conversation database instance."""
    return ConversationDB()
