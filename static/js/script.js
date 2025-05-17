// script.js
document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingContainer = document.getElementById('loading-container');
    const typingIndicator = document.createElement('div');
    typingIndicator.classList.add('typing-indicator');
    typingIndicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

    // Function to add a message to the chat container
    function addMessage(content, type, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${type}-message`);
        
        // Add avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.classList.add('message-avatar', `${type}-avatar`);
        
        // Different icon for user vs system
        if (type === 'user') {
            avatarDiv.innerHTML = '<i class="fas fa-user"></i>';
        } else {
            avatarDiv.innerHTML = '<i class="fas fa-robot"></i>';
        }
        
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        
        const paragraph = document.createElement('p');
        paragraph.classList.add('message-text');
        paragraph.textContent = content;
        messageContent.appendChild(paragraph);
        
        // Add sources if available
        if (sources && sources.length > 0) {
            const sourceList = document.createElement('div');
            sourceList.classList.add('source-list');
            
            const sourceTitle = document.createElement('h4');
            sourceTitle.innerHTML = '<i class="fas fa-link"></i> Sources:';
            sourceList.appendChild(sourceTitle);
            
            const sourcesContainer = document.createElement('div');
            sourcesContainer.classList.add('sources-container');
            
            sources.forEach(source => {
                const sourceItem = document.createElement('div');
                sourceItem.classList.add('source-item');
                
                // Create favicon element
                const favicon = document.createElement('img');
                favicon.classList.add('source-favicon');
                
                // Extract domain for favicon
                let domain;
                try {
                    domain = new URL(source).hostname;
                } catch (e) {
                    domain = "unknown";
                }
                
                // Set favicon URL using Google's favicon service
                favicon.src = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
                favicon.onerror = function() {
                    // Fallback to a generic icon if favicon fails to load
                    this.onerror = null;
                    this.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16"><path fill="%234361ee" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 15h2v-6h-2v6zm0-8h2V7h-2v2z"/></svg>';
                };
                
                const sourceLink = document.createElement('a');
                sourceLink.href = source;
                
                // Format the URL for display
                try {
                    const url = new URL(source);
                    let displayText = url.hostname + url.pathname;
                    // Truncate if too long
                    if (displayText.length > 40) {
                        displayText = displayText.substring(0, 37) + '...';
                    }
                    sourceLink.textContent = displayText;
                } catch (e) {
                    sourceLink.textContent = source;
                }
                sourceLink.target = "_blank";
                sourceLink.classList.add('source-link');
                sourceLink.rel = "noopener noreferrer"; // Security best practice
                
                sourceItem.appendChild(favicon);
                sourceItem.appendChild(sourceLink);
                sourcesContainer.appendChild(sourceItem);
            });
            
            sourceList.appendChild(sourcesContainer);
            messageContent.appendChild(sourceList);
        }
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(messageContent);
        chatContainer.appendChild(messageDiv);
        
        // Scroll to the bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Show typing indicator
    function showTypingIndicator() {
        chatContainer.appendChild(typingIndicator);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Hide typing indicator
    function hideTypingIndicator() {
        if (typingIndicator.parentNode === chatContainer) {
            chatContainer.removeChild(typingIndicator);
        }
    }

    // Function to send user question and get response
    async function sendQuestion(question) {
        // Disable input and show loading
        userInput.disabled = true;
        sendButton.disabled = true;
        loadingContainer.style.display = 'flex';
        
        // Show typing indicator
        showTypingIndicator();
        
        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question })
            });
            
            const data = await response.json();
            
            // Hide typing indicator
            hideTypingIndicator();
            
            if (response.ok) {
                // Add the AI response to the chat
                addMessage(data.answer, 'system', data.sources);
            } else {
                // Handle errors
                addMessage(`Error: ${data.error || 'Failed to get response'}`, 'system');
            }
        } catch (error) {
            console.error('Error:', error);
            // Hide typing indicator
            hideTypingIndicator();
            addMessage('Sorry, there was an error processing your request. Please try again later.', 'system');
        } finally {
            // Re-enable input and hide loading
            userInput.disabled = false;
            sendButton.disabled = false;
            loadingContainer.style.display = 'none';
            userInput.focus();
        }
    }

    // Event listener for send button
    sendButton.addEventListener('click', function() {
        const question = userInput.value.trim();
        if (question) {
            // Add user message to chat
            addMessage(question, 'user');
            // Clear input
            userInput.value = '';
            // Send question to server
            sendQuestion(question);
        }
    });

    // Event listener for Enter key
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendButton.click();
        }
    });

    // Auto-focus the input field on page load
    userInput.focus();

    // Auto-resize textarea as user types
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        // Limit max height to avoid filling the entire screen
        const maxHeight = 150;
        const newHeight = Math.min(this.scrollHeight, maxHeight);
        this.style.height = newHeight + 'px';
    });
    
    // Add welcome message
    addMessage('Hello! Im your Web RAG assistant. Ask me any question and Ill search the internet to find you the best answer.', 'system');
});