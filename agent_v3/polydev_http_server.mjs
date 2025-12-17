#!/usr/bin/env node
/**
 * Polydev HTTP Server Wrapper
 *
 * Exposes the Polydev MCP as an HTTP API for the Python SWE-bench agent.
 *
 * Usage:
 *   Start server: node polydev_http_server.mjs
 *   Call API: POST http://localhost:3847/perspectives {"prompt": "..."}
 */

import { createServer } from 'http';
import { spawn } from 'child_process';
import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

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
const POLYDEV_TOKEN = process.env.POLYDEV_USER_TOKEN;

if (!POLYDEV_TOKEN) {
  console.error('ERROR: POLYDEV_USER_TOKEN not found in environment');
  process.exit(1);
}

// MCP server state
let mcpProcess = null;
let mcpReady = false;
let requestId = 0;
let pendingRequests = new Map();

// Start the Polydev MCP server
function startMCPServer() {
  console.log('Starting Polydev MCP server...');

  mcpProcess = spawn('npx', ['-y', 'polydev-ai@latest'], {
    env: {
      ...process.env,
      POLYDEV_USER_TOKEN: POLYDEV_TOKEN,
    },
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  let buffer = '';

  mcpProcess.stdout.on('data', (data) => {
    buffer += data.toString();

    // Process complete JSON-RPC messages
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const message = JSON.parse(line);
        handleMCPResponse(message);
      } catch (e) {
        // Not JSON, might be debug output
        if (line.includes('ready') || line.includes('initialized')) {
          mcpReady = true;
          console.log('Polydev MCP server ready');
        }
      }
    }
  });

  mcpProcess.stderr.on('data', (data) => {
    const msg = data.toString().trim();
    if (msg) console.error('MCP stderr:', msg);
    // Some servers output ready message to stderr
    if (msg.includes('ready') || msg.includes('listening')) {
      mcpReady = true;
    }
  });

  mcpProcess.on('close', (code) => {
    console.log(`MCP server exited with code ${code}`);
    mcpReady = false;
    // Restart after delay
    setTimeout(startMCPServer, 1000);
  });

  // Initialize the MCP server
  setTimeout(() => {
    sendMCPRequest('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'polydev-http-wrapper', version: '1.0.0' }
    });
  }, 500);
}

// Send JSON-RPC request to MCP server
function sendMCPRequest(method, params) {
  const id = ++requestId;
  const request = {
    jsonrpc: '2.0',
    id,
    method,
    params,
  };

  return new Promise((resolve, reject) => {
    pendingRequests.set(id, { resolve, reject, timestamp: Date.now() });
    mcpProcess.stdin.write(JSON.stringify(request) + '\n');

    // Timeout after 120s
    setTimeout(() => {
      if (pendingRequests.has(id)) {
        pendingRequests.delete(id);
        reject(new Error('MCP request timeout'));
      }
    }, 120000);
  });
}

// Handle MCP response
function handleMCPResponse(message) {
  if (message.id && pendingRequests.has(message.id)) {
    const { resolve, reject } = pendingRequests.get(message.id);
    pendingRequests.delete(message.id);

    if (message.error) {
      reject(new Error(message.error.message || 'MCP error'));
    } else {
      resolve(message.result);
    }

    // Mark as ready after successful initialize
    if (!mcpReady && message.result) {
      mcpReady = true;
      console.log('Polydev MCP initialized successfully');
    }
  }
}

// Call Polydev get_perspectives tool
async function getPerspectives(prompt, models) {
  if (!mcpReady) {
    throw new Error('MCP server not ready');
  }

  const result = await sendMCPRequest('tools/call', {
    name: 'get_perspectives',
    arguments: {
      prompt,
      models: models || ['gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-2.0-flash-exp'],
    },
  });

  return result;
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
      status: mcpReady ? 'ready' : 'starting',
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
        const { prompt, models } = JSON.parse(body);

        if (!prompt) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'prompt is required' }));
          return;
        }

        console.log(`[${new Date().toISOString()}] Getting perspectives for: ${prompt.substring(0, 100)}...`);

        const startTime = Date.now();
        const result = await getPerspectives(prompt, models);
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

  // 404 for other routes
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

// Start everything
startMCPServer();

server.listen(PORT, () => {
  console.log(`Polydev HTTP server listening on http://localhost:${PORT}`);
  console.log('Endpoints:');
  console.log('  GET  /health       - Health check');
  console.log('  POST /perspectives - Get AI perspectives');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  if (mcpProcess) mcpProcess.kill();
  server.close();
  process.exit(0);
});
