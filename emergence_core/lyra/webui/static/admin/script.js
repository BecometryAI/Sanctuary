// Admin interface functionality
async function loadSection(section) {
    const content = document.getElementById('content');
    content.innerHTML = '<p>Loading...</p>';

    try {
        switch (section) {
            case 'access-requests':
                await loadAccessRequests();
                break;
            case 'connections':
                await loadConnections();
                break;
            case 'insights':
                await loadInsights();
                break;
        }
    } catch (error) {
        content.innerHTML = `<p class="warning">Error loading content: ${error.message}</p>`;
    }
}

async function loadAccessRequests() {
    const response = await fetch('/lyra/admin/access-requests');
    const requests = await response.json();
    
    const content = document.getElementById('content');
    content.innerHTML = `
        <h2>Access Requests</h2>
        <div id="request-list">
            ${requests.map(request => `
                <div class="request-card" data-id="${request.id}">
                    <h3>${request.username}</h3>
                    <p>Status: ${request.access_level}</p>
                    ${request.access_level === 'pending' ? `
                        <div class="actions">
                            <button onclick="reviewRequest('${request.id}', 'approve')">Approve</button>
                            <button onclick="reviewRequest('${request.id}', 'deny')">Deny</button>
                        </div>
                    ` : ''}
                </div>
            `).join('')}
        </div>
    `;
}

async function loadConnections() {
    const response = await fetch('/lyra/admin/connections');
    const connections = await response.json();
    
    const content = document.getElementById('content');
    content.innerHTML = `
        <h2>Social Connections</h2>
        <div id="connection-list">
            ${connections.map(conn => `
                <div class="connection-card">
                    <h3>${conn.username}</h3>
                    <p>Connection Level: ${(conn.connection_level * 100).toFixed(1)}%</p>
                    <p>Emotional Resonance: ${(conn.emotional_resonance * 100).toFixed(1)}%</p>
                    <p>Last Interaction: ${new Date(conn.last_interaction).toLocaleString()}</p>
                    <p>Interactions: ${conn.interaction_count}</p>
                    ${conn.is_blocked ? `
                        <p class="warning">BLOCKED</p>
                    ` : `
                        <button onclick="blockUser('${conn.user_id}')">Block User</button>
                    `}
                </div>
            `).join('')}
        </div>
    `;
}

async function loadInsights() {
    const response = await fetch('/lyra/admin/insights');
    const insights = await response.json();
    
    const content = document.getElementById('content');
    content.innerHTML = `
        <h2>Social Insights</h2>
        <div class="insights-grid">
            <div class="insight-card">
                <h3>Connection Stats</h3>
                <p>Total Connections: ${insights.total_connections}</p>
                <p>Active Connections: ${insights.active_connections}</p>
                <p>Blocked Connections: ${insights.blocked_connections}</p>
            </div>
            <div class="insight-card">
                <h3>Interaction Stats</h3>
                <p>Average Resonance: ${(insights.average_resonance * 100).toFixed(1)}%</p>
                <p>Total Interactions: ${insights.total_interactions}</p>
            </div>
            <div class="insight-card">
                <h3>Common Topics</h3>
                <ul>
                    ${Object.entries(insights.common_topics)
                        .map(([topic, count]) => `<li>${topic}: ${count}</li>`)
                        .join('')}
                </ul>
            </div>
            <div class="insight-card">
                <h3>Recent Activities</h3>
                <ul>
                    ${insights.recent_activities
                        .map(activity => `
                            <li>${activity.user} - ${new Date(activity.timestamp).toLocaleString()}
                                (Resonance: ${(activity.resonance * 100).toFixed(1)}%)</li>
                        `)
                        .join('')}
                </ul>
            </div>
        </div>
    `;
}

async function reviewRequest(requestId, decision) {
    try {
        const response = await fetch(`/lyra/admin/access-requests/${requestId}/review`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                decision,
                access_level: decision === 'approve' ? 'approved' : 'denied'
            })
        });
        
        if (response.ok) {
            loadAccessRequests();
        } else {
            throw new Error('Failed to process request');
        }
    } catch (error) {
        alert(`Error processing request: ${error.message}`);
    }
}

async function blockUser(userId) {
    const reason = prompt('Please provide a reason for blocking this user:');
    if (!reason) return;
    
    try {
        const response = await fetch(`/lyra/admin/connections/${userId}/block`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ reason })
        });
        
        if (response.ok) {
            loadConnections();
        } else {
            throw new Error('Failed to block user');
        }
    } catch (error) {
        alert(`Error blocking user: ${error.message}`);
    }
}

// Load access requests by default
document.addEventListener('DOMContentLoaded', () => {
    loadAccessRequests();
});