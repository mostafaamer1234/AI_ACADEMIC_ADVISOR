import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { generateResponse, setupEnvironment, clearConversation } from '../services/apiService';
import MarkdownRenderer from './MarkdownRenderer';
import '../styles/UnifiedChat.css';

const UnifiedChat = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [transcriptUploaded, setTranscriptUploaded] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [conversationHistory, setConversationHistory] = useState([]);
    const chatEndRef = useRef(null);
    const fileInputRef = useRef(null);
    const navigate = useNavigate();
    
    // Create or reuse a per-tab session id for isolation
    const getSessionId = () => {
        let sessionId = sessionStorage.getItem('session_id');
        if (!sessionId) {
            sessionId = typeof crypto !== 'undefined' && crypto.randomUUID
                ? crypto.randomUUID()
                : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
            sessionStorage.setItem('session_id', sessionId);
        }
        return sessionId;
    };

    // Load conversation history from sessionStorage on mount
    useEffect(() => {
        const sessionId = getSessionId();
        const savedHistory = sessionStorage.getItem(`chat_history_${sessionId}`);
        const savedMessages = sessionStorage.getItem(`chat_messages_${sessionId}`);
        const savedTranscriptFlag = sessionStorage.getItem(`transcript_uploaded_${sessionId}`);
        if (savedHistory) {
            try {
                setConversationHistory(JSON.parse(savedHistory));
            } catch (e) {
                console.error('Error loading conversation history:', e);
            }
        }
        if (savedMessages) {
            try {
                const parsedMessages = JSON.parse(savedMessages);
                // Convert timestamp strings back to Date objects
                const messagesWithDates = parsedMessages.map(msg => ({
                    ...msg,
                    timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date()
                }));
                setMessages(messagesWithDates);
            } catch (e) {
                console.error('Error loading messages:', e);
            }
        }
        if (savedTranscriptFlag === 'true') {
            setTranscriptUploaded(true);
        }
        
        // Cleanup function - clear chat when component unmounts (user navigates away)
        return () => {
            // Don't clear on unmount, only on explicit logout
            // The cleanup is handled in handleLogout
        };
    }, []);

    // Save conversation history to sessionStorage
    useEffect(() => {
        const sessionId = getSessionId();
        if (conversationHistory.length > 0) {
            sessionStorage.setItem(`chat_history_${sessionId}`, JSON.stringify(conversationHistory));
        }
        if (messages.length > 0) {
            sessionStorage.setItem(`chat_messages_${sessionId}`, JSON.stringify(messages));
        }
        sessionStorage.setItem(`transcript_uploaded_${sessionId}`, transcriptUploaded ? 'true' : 'false');
    }, [conversationHistory, messages, transcriptUploaded]);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setIsUploading(true);
        setError('');
        
        try {
            const formData = new FormData();
            formData.append('transcript', file);
            
            const sessionId = getSessionId();
            const message = await setupEnvironment(formData, sessionId);
            console.log('Environment initialized:', message);
            
            setTranscriptUploaded(true);
            
            // Add system message about transcript upload
            const uploadMessage = {
                id: Date.now(),
                role: 'assistant',
                content: '✅ Your transcript has been uploaded successfully! I can now provide personalized course planning, graduation plans, and answer questions specific to your academic progress. What would you like to know?',
                timestamp: new Date()
            };
            
            setMessages(prev => [...prev, uploadMessage]);
            
        } catch (error) {
            console.error('Error uploading transcript:', error);
            setError('Failed to upload transcript. Please try again.');
        } finally {
            setIsUploading(false);
            // Reset file input
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    const handleSendMessage = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage = {
            id: Date.now(),
            role: 'user',
            content: input.trim(),
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setError('');
        setIsLoading(true);

        try {
            const result = await generateResponse(
                userMessage.content,
                conversationHistory,
                getSessionId()
            );

            const aiMessage = {
                id: Date.now() + 1,
                role: 'assistant',
                content: result.response,
                timestamp: new Date()
            };

            setMessages(prev => [...prev, aiMessage]);
            setConversationHistory(result.conversationHistory || []);
        } catch (error) {
            console.error('Error sending message:', error);
            setError('Failed to get a response. Please try again.');
            
            // Add error message to chat
            const errorMessage = {
                id: Date.now() + 1,
                role: 'assistant',
                content: 'Sorry, I encountered an error. Please try again.',
                timestamp: new Date(),
                isError: true
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    const handleClearChat = async () => {
        if (window.confirm('Are you sure you want to clear the conversation?')) {
            try {
                const sessionId = getSessionId();
                await clearConversation(sessionId);
                setMessages([]);
                setConversationHistory([]);
                setTranscriptUploaded(false);
                sessionStorage.removeItem(`chat_history_${sessionId}`);
                sessionStorage.removeItem(`chat_messages_${sessionId}`);
                sessionStorage.removeItem(`transcript_uploaded_${sessionId}`);
            } catch (error) {
                console.error('Error clearing conversation:', error);
            }
        }
    };

    return (
        <div className="unified-chat-container">
            <div className="chat-header">
                <h2>AI Academic Advisor</h2>
                <div className="chat-header-actions">
                    <button
                        onClick={() => navigate('/faculty-dashboard')}
                        className="course-graph-button"
                    >
                        📈 Course Graph
                    </button>
                    <label className="upload-button">
                        {transcriptUploaded ? '✅ Transcript Uploaded' : '📄 Upload Transcript'}
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".pdf"
                            onChange={handleFileUpload}
                            style={{ display: 'none' }}
                            disabled={isUploading}
                        />
                    </label>
                    {messages.length > 0 && (
                        <button onClick={handleClearChat} className="clear-button">
                            🗑️ Clear Chat
                        </button>
                    )}
                </div>
            </div>

            {messages.length === 0 && (
                <div className="welcome-message">
                    <h3>Welcome! 👋</h3>
                    <p>I'm your AI Academic Advisor. Feel free to ask me anything about:</p>
                    <ul>
                        <li>Course information and prerequisites</li>
                        <li>Major requirements</li>
                        <li>General academic advice</li>
                        <li>Graduation planning (better with transcript)</li>
                        <li>Course recommendations</li>
                    </ul>
                    <p style={{ marginTop: '15px', fontStyle: 'italic', color: '#666' }}>
                        💡 Tip: Upload your transcript for personalized graduation planning and course recommendations based on your academic history.
                    </p>
                </div>
            )}

            <div className="chat-messages">
                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`message ${message.role} ${message.isError ? 'error' : ''}`}
                    >
                        <div className="message-content">
                            {message.role === 'assistant' && !message.isError ? (
                                <MarkdownRenderer content={message.content} />
                            ) : (
                                message.content
                            )}
                        </div>
                        <div className="message-timestamp">
                            {message.timestamp instanceof Date 
                                ? message.timestamp.toLocaleTimeString() 
                                : new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className="message assistant loading">
                        <div className="message-content">
                            <span className="typing-indicator">
                                <span></span>
                                <span></span>
                                <span></span>
                            </span>
                        </div>
                    </div>
                )}
                <div ref={chatEndRef} />
            </div>

            {error && <div className="error-message">{error}</div>}

            <div className="chat-input-container">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={transcriptUploaded 
                        ? "Ask me anything about your courses, graduation plan, or academic requirements..." 
                        : "Ask me anything! (Upload transcript for personalized planning)..."}
                    disabled={isLoading || isUploading}
                    className="chat-input"
                />
                <button
                    onClick={handleSendMessage}
                    disabled={isLoading || isUploading || !input.trim()}
                    className="send-button"
                >
                    {isLoading ? '⏳' : '📤'}
                </button>
            </div>

            {isUploading && (
                <div className="upload-status">📤 Uploading transcript...</div>
            )}
        </div>
    );
};

export default UnifiedChat;

