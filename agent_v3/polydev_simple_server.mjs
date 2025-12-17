#!/usr/bin/env node
/**
 * Simple Polydev HTTP Server
 *
 * Uses the mcp-execution module directly to call Polydev.
 * This avoids having to implement the MCP protocol manually.
 *
 * Usage:
 *   Start server: node polydev_simple_server.mjs
 *   Call API: POST http://localhost:3847/perspectives {"prompt": "..."}
 */

import { createServer } from 'http';
import { readFileSync } from 'fs';

// Load env from mcp-execution .env file
const envPath = '/Users/venkat/mcp-execution/.env';
try {
  const envContent = readFileSync(envPath, 'utf8');
  for (const line of envContent.split('\n')) {
    const match = line.match(/^([^#=]+)=(.*)$/);
    if (match) {
      const [, key, value] = match;
      if (!process.env[key.trim()]) {
        process.env[key.trim()] = value.trim();
      }
    }
  }
} catch (e) {
  console.error('Warning: Could not load .env file:', e.message);
}

const PORT = process.env.POLYDEV_HTTP_PORT || 3847;

// Import polydev from mcp-execution
let polydev;
try {
  const mcpExecution = await import('/Users/venkat/mcp-execution/dist/index.js');
  polydev = mcpExecution.polydev;
  console.log('Polydev module loaded successfully');
} catch (e) {
  console.error('Failed to load mcp-execution:', e.message);
  process.exit(1);
}

// Initialize polydev
let initialized = false;
async function ensureInitialized() {
  if (!initialized) {
    console.log('Initializing Polydev...');
    await polydev.initialize();
    initialized = true;
    console.log('Polydev initialized');
  }
}

// HTTP server
const server = createServer(async (req, res) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // Health check
  if (req.url === '/health' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status: initialized ? 'ready' : 'starting',
      uptime: process.uptime(),
    }));
    return;
  }

  // Get perspectives endpoint
  if (req.url === '/perspectives' && req.method === 'POST') {
    let body = '';

    req.on('data', chunk => { body += chunk; });

    req.on('end', async () => {
      try {
        const { prompt, models, systemPrompt } = JSON.parse(body);

        if (!prompt) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'prompt is required' }));
          return;
        }

        await ensureInitialized();

        console.log(`[${new Date().toISOString()}] Getting perspectives for: ${prompt.substring(0, 100)}...`);

        const startTime = Date.now();
        const result = await polydev.getPerspectives(prompt, models, systemPrompt);
        const latency = Date.now() - startTime;

        console.log(`[${new Date().toISOString()}] Perspectives received in ${latency}ms`);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          latency_ms: latency,
          result,
        }));

      } catch (error) {
        console.error('Error getting perspectives:', error.message);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: false,
          error: error.message,
        }));
      }
    });

    return;
  }

  // List models endpoint
  if (req.url === '/models' && req.method === 'GET') {
    try {
      await ensureInitialized();
      const models = await polydev.listAvailableModels();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ models }));
    } catch (error) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message }));
    }
    return;
  }

  // 404 for other routes
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

// Start server
server.listen(PORT, () => {
  console.log(`Polydev HTTP server listening on http://localhost:${PORT}`);
  console.log('Endpoints:');
  console.log('  GET  /health       - Health check');
  console.log('  GET  /models       - List available models');
  console.log('  POST /perspectives - Get AI perspectives');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  server.close();
  process.exit(0);
});
