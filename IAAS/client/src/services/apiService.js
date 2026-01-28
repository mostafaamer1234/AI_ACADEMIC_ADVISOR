const API_URL = 'http://localhost:5002/llm'; // Flask backend URL

export const generateResponse = async (userQuestion, conversationHistory = [], userId = 'default') => {
    try {
        const response = await fetch("http://127.0.0.1:5002/llm/generate-response", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                user_question: userQuestion,
                conversation_history: conversationHistory,
                user_id: userId
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Response from LLM:", data);
        return {
            response: data.graduation_plan || data.response || '',
            conversationHistory: data.conversation_history || conversationHistory
        };
    } catch (error) {
        console.error('Error querying LLM:', error);
        throw error;
    }
};

export const clearConversation = async (userId = 'default') => {
    try {
        const response = await fetch("http://127.0.0.1:5002/llm/clear-conversation", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_id: userId }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error clearing conversation:', error);
        throw error;
    }
};

export const setupEnvironment = async (formData, userId = 'default') => {
    try {
        // Append user_id to formData
        formData.append('user_id', userId);
        
        const response = await fetch(`http://127.0.0.1:5002/llm/setup-environment`, {
            method: 'POST',
            body: formData, // Send the FormData object
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data.message; // Return the success message from the backend
    } catch (error) {
        console.error('Error setting up environment:', error);
        throw error;
    }
};
