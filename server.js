// server.js - Gemini RAG Bot
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();

// ---------- MIDDLEWARE ----------
app.use(
  cors({
    origin: [
      'http://localhost:5173', // Vite dev (change/extend if needed)
      'http://localhost:3000',
      // 'https://your-frontend-domain.com', // add your prod domain later
    ],
    credentials: true,
  })
);
app.use(express.json());

// ---------- GEMINI SETUP ----------
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
// use a stable, available model alias
const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

// ---------- POSTGRES / SUPABASE SETUP ----------
/*
  In your .env:

  DATABASE_URL=postgres://postgres:YOUR_PASSWORD@db.xxxxxx.supabase.co:5432/postgres
*/
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: {
    rejectUnauthorized: false, // required for Supabase
  },
});

// ---------- DB SCHEMA DESCRIPTION (for Gemini prompt) ----------
const DB_SCHEMA = `
You are connected to a PostgreSQL database for supply chain analytics.

Tables and columns:

1) inventory
   - id (serial, primary key)
   - sku (text)
   - product_name (text)
   - location (text)
   - quantity (integer)
   - safety_stock (integer)
   - updated_at (timestamp)

2) orders
   - id (serial, primary key)
   - order_date (date)
   - customer_name (text)
   - status (text) -- 'pending', 'shipped', 'delayed'
   - total_amount (numeric)

3) shipments
   - id (serial, primary key)
   - shipment_date (date)
   - origin (text)
   - destination (text)
   - status (text) -- 'on-time', 'delayed'
   - carrier (text)
   - tracking_number (text)

4) suppliers
   - id (serial, primary key)
   - supplier_name (text)
   - lead_time_days (integer)
   - on_time_rate (numeric) -- 0-1 fraction

Use ONLY these tables and columns when writing SQL.
`;

// ---------- SQL PLANNER (Gemini) ----------
async function generateSQL(userQuery) {
  const prompt = `
You are a SQL planner for a supply chain PostgreSQL database.

Database Schema:
${DB_SCHEMA}

User Request: "${userQuery}"

Your job:
1. Decide if we need to run a SQL query.
   - If the user asks for actual data, numbers, counts, lists, dates, etc. -> "use_sql": true
   - If the user asks for general theory or definitions (e.g. "What is safety stock?") -> "use_sql": false

2. If "use_sql" = true:
   - Write ONE valid PostgreSQL SELECT query.
   - Use ONLY the tables/columns from the schema.
   - Never use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, etc.
   - Use a LIMIT when returning many rows (e.g. LIMIT 100).

3. Respond with ONLY raw JSON. No markdown, no backticks, no explanation.

Format (very important):
{
  "use_sql": true or false,
  "sql": "SELECT ...",   // a string, or null if use_sql is false
  "reason": "short explanation"
}
`;

  const result = await model.generateContent(prompt);
  let content = result.response.text().trim();

  console.log('Raw Gemini plan:', content);

  // Strip ```json ... ``` if Gemini decides to be cute
  if (content.startsWith('```')) {
    content = content.replace(/```json/gi, '').replace(/```/g, '').trim();
  }

  // Extract first {...} block
  const match = content.match(/\{[\s\S]*\}/);
  if (!match) {
    console.error('No JSON object found in Gemini response');
    return { use_sql: false, sql: null, reason: 'No JSON object in model response' };
  }

  try {
    return JSON.parse(match[0]);
  } catch (e) {
    console.error('Failed to parse JSON from Gemini:', e.message);
    return { use_sql: false, sql: null, reason: 'JSON Parse Error: ' + e.message };
  }
}

// ---------- FINAL ANSWER GENERATION (Gemini) ----------
async function generateFinalAnswer(userQuery, dbData, plan) {
  const prompt = `
You are a Supply Chain AI Assistant.

User question:
${userQuery}

SQL plan (for context):
${JSON.stringify(plan, null, 2)}

Database result rows (if any):
${JSON.stringify(dbData, null, 2)}

Instructions:
- If "use_sql" is true and rows exist, use them as ground truth.
- Explain clearly and simply, like you're talking to a supply chain analyst.
- Mention important numbers, trends, or counts.
- If no rows were returned, honestly say there is no data for that query.
- If "use_sql" is false, answer from general supply chain knowledge (no fake numbers).
`;

  const result = await model.generateContent(prompt);
  return result.response.text();
}

// ---------- /chat ROUTE ----------
app.post('/chat', async (req, res) => {
  const { message } = req.body;

  if (!message || !message.trim()) {
    return res.status(400).json({ error: 'message is required' });
  }

  try {
    // 1) Plan SQL
    const plan = await generateSQL(message);
    console.log('SQL Plan:', plan);

    let rows = [];

    // 2) Run SQL if requested
    if (plan.use_sql && plan.sql) {
      try {
        console.log('Executing SQL:', plan.sql);
        const result = await pool.query(plan.sql);
        rows = result.rows;
      } catch (err) {
        console.error('SQL error:', err);
        return res.status(400).json({
          error: 'SQL Failed',
          detail: err.message,
          sql: plan.sql,
        });
      }
    }

    // 3) Generate final natural language answer
    const finalAnswer = await generateFinalAnswer(message, rows, plan);

    return res.json({
      summary: finalAnswer, // your frontend uses this
      sql: plan.sql || null,
      rows,                 // nice for debugging
      plan,                 // optional: can inspect in dev tools
    });
  } catch (err) {
    console.error('Server/Gemini error:', err);
    return res.status(500).json({
      error: 'Internal Error',
      details: err.message,
    });
  }
});

// ---------- HEALTH CHECK ----------
app.get('/', (req, res) => res.send('Gemini RAG Bot Running'));

// ---------- START SERVER ----------
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Server ON ${PORT}`));
