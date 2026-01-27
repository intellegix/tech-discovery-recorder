const express = require('express');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

// Services to keep alive
const SERVICES = [
    {
        name: 'Frontend',
        url: 'https://tech-discovery-recorder.onrender.com',
        enabled: true
    },
    {
        name: 'Backend API',
        url: 'https://tech-discovery-recorder-api.onrender.com/health',
        enabled: true
    }
];

// Ping function
function pingService(service) {
    return new Promise((resolve, reject) => {
        const req = https.get(service.url, (res) => {
            const statusCode = res.statusCode;
            console.log(`[${new Date().toISOString()}] ${service.name}: ${statusCode} ${statusCode < 400 ? '✅' : '❌'}`);
            resolve({ service: service.name, statusCode, success: statusCode < 400 });
        });

        req.on('error', (err) => {
            console.log(`[${new Date().toISOString()}] ${service.name}: ERROR - ${err.message} ❌`);
            resolve({ service: service.name, error: err.message, success: false });
        });

        req.setTimeout(30000, () => {
            req.abort();
            console.log(`[${new Date().toISOString()}] ${service.name}: TIMEOUT ⏱️`);
            resolve({ service: service.name, error: 'timeout', success: false });
        });
    });
}

// Ping all services
async function pingAllServices() {
    console.log(`\n🔄 [${new Date().toISOString()}] Starting ping cycle...`);

    const results = [];
    for (const service of SERVICES) {
        if (service.enabled) {
            const result = await pingService(service);
            results.push(result);
        }
    }

    const successCount = results.filter(r => r.success).length;
    console.log(`📊 Ping complete: ${successCount}/${results.length} services responding\n`);

    return results;
}

// Health endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        purpose: 'auto-ping service to prevent Render apps from sleeping'
    });
});

// Status endpoint
app.get('/status', async (req, res) => {
    const results = await pingAllServices();
    res.json({
        timestamp: new Date().toISOString(),
        services: results
    });
});

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        name: 'Tech Discovery Recorder Auto-Ping Service',
        purpose: 'Keeps frontend and backend applications awake on Render',
        endpoints: [
            '/health - Service health check',
            '/status - Ping all services and return status'
        ],
        services: SERVICES.filter(s => s.enabled).map(s => ({ name: s.name, url: s.url }))
    });
});

// Start server
app.listen(PORT, () => {
    console.log('🚀 Tech Discovery Auto-Ping Service Started');
    console.log(`📡 Server running on port ${PORT}`);
    console.log('🎯 Purpose: Keep Render apps awake');
    console.log('\n📋 Monitoring Services:');
    SERVICES.filter(s => s.enabled).forEach(service => {
        console.log(`   • ${service.name}: ${service.url}`);
    });
    console.log('\n⏰ Ping schedule: Every 10 minutes');

    // Initial ping
    setTimeout(pingAllServices, 5000);

    // Set up regular pings every 10 minutes (600000 ms)
    setInterval(pingAllServices, 600000);
});
