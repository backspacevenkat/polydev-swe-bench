#!/usr/bin/env node
/**
 * Polydev MCP Consultation Script
 *
 * Calls Polydev to get multi-model perspectives on a problem.
 * Used by the SWE-bench agent for consultation when stuck.
 *
 * Usage: node polydev_consultation.mjs "Your problem description"
 * Output: JSON with perspectives from multiple models
 */

import { spawn } from 'child_process';

async function getLocalCLIPerspectives(prompt) {
  const perspectives = [];

  // Query GPT via codex CLI (this works well)
  const codexPromise = new Promise((resolve) => {
    const startTime = Date.now();
    const proc = spawn('codex', ['exec', '--json', prompt], {
      timeout: 120000,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let output = '';
    let error = '';

    proc.stdout.on('data', (data) => { output += data; });
    proc.stderr.on('data', (data) => { error += data; });

    proc.on('close', (code) => {
      const latency = Date.now() - startTime;
      let response = '';
      let tokens = 0;

      // Parse JSONL output from codex
      try {
        for (const line of output.split('\n')) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          if (event.type === 'item.completed' && event.item?.type === 'agent_message') {
            response = event.item.text || '';
          }
          if (event.type === 'turn.completed') {
            tokens = (event.usage?.input_tokens || 0) + (event.usage?.output_tokens || 0);
          }
        }
      } catch (e) {
        // If not JSONL, use raw output
        response = output.trim() || error.trim();
        tokens = Math.ceil(response.length / 4);
      }

      resolve({
        model: 'gpt-5.2',
        response: response || 'No response from Codex',
        latency_ms: latency,
        tokens_estimate: tokens || Math.ceil((response || '').length / 4)
      });
    });

    proc.on('error', (err) => {
      resolve({
        model: 'gpt-5.2',
        response: `Codex CLI error: ${err.message}`,
        latency_ms: 0,
        tokens_estimate: 0
      });
    });

    // Set timeout
    setTimeout(() => {
      proc.kill();
    }, 120000);
  });

  // Query Claude via claude CLI with simpler approach
  const claudePromise = new Promise((resolve) => {
    const startTime = Date.now();

    // Use echo pipe approach for more reliable input
    const proc = spawn('claude', ['--output-format', 'text'], {
      timeout: 60000,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let output = '';
    let error = '';

    proc.stdout.on('data', (data) => { output += data; });
    proc.stderr.on('data', (data) => { error += data; });

    // Write prompt to stdin
    proc.stdin.write(prompt);
    proc.stdin.end();

    proc.on('close', (code) => {
      const latency = Date.now() - startTime;
      const response = code === 0 && output.trim() ? output.trim() : (error.trim() || 'No response');

      resolve({
        model: 'claude-sonnet-4',
        response: response,
        latency_ms: latency,
        tokens_estimate: Math.ceil(response.length / 4)
      });
    });

    proc.on('error', (err) => {
      resolve({
        model: 'claude-sonnet-4',
        response: `Claude CLI error: ${err.message}`,
        latency_ms: 0,
        tokens_estimate: 0
      });
    });

    // Set timeout
    setTimeout(() => {
      proc.kill();
    }, 60000);
  });

  // Run queries in parallel
  const [codex, claude] = await Promise.all([codexPromise, claudePromise]);

  return {
    perspectives: [codex, claude],
    total_latency_ms: Math.max(codex.latency_ms, claude.latency_ms),
    total_tokens: codex.tokens_estimate + claude.tokens_estimate
  };
}

// Main
const prompt = process.argv[2];

if (!prompt) {
  console.error('Usage: node polydev_consultation.mjs "Your problem description"');
  process.exit(1);
}

try {
  const result = await getLocalCLIPerspectives(prompt);
  console.log(JSON.stringify(result, null, 2));
} catch (e) {
  console.error(JSON.stringify({ error: e.message }));
  process.exit(1);
}
