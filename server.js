// server.js — Gemini RAG with Supabase + Hugging Face embeddings

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { createClient } = require('@supabase/supabase-js');
const { HfInference } = require('@huggingface/inference');


const app = express();

/* -------------------- MIDDLEWARE -------------------- */
app.use(
  cors({
    origin: ['http://localhost:5174', 'http://localhost:3000'],
    credentials: true,
  })
);
app.use(express.json());

/* -------------------- GEMINI (CHAT) -------------------- */
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const chatModel = genAI.getGenerativeModel({
  model: 'gemini-2.5-flash',
});

/* -------------------- HUGGING FACE (EMBEDDINGS) -------------------- */
const HF_EMBEDDING_MODEL =
  process.env.HF_EMBEDDING_MODEL ||
  'sentence-transformers/all-MiniLM-L6-v2';
const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);


/* -------------------- SUPABASE -------------------- */
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY,
  { auth: { autoRefreshToken: false, persistSession: false } }
);

/* -------------------- HELPERS -------------------- */

// Embed text using Hugging Face
async function embedText(text) {
  const embedding = await hf.featureExtraction({
    model: 'sentence-transformers/all-MiniLM-L6-v2',
    inputs: text,
  });

  if (!Array.isArray(embedding)) {
    throw new Error('Invalid embedding returned from Hugging Face');
  }

  return embedding; // length = 384
}



// Insert / update document
async function upsertDocument({ id, title, text, metadata }) {
  const embedding = await embedText(text);

  const { data, error } = await supabase
    .from('documents')
    .upsert(
      {
        id,
        title: title || null,
        content: text,
        metadata: metadata || {},
        embedding,
      },
      { onConflict: 'id' }
    )
    .select('*');

  if (error) throw error;
  return data;
}

// Vector similarity search
async function searchSimilar(
  embedding,
  matchCount = 6,
  similarityThreshold = 0.6
) {
  const { data, error } = await supabase.rpc('match_documents', {
    query_embedding: embedding,
    match_count: matchCount,
    similarity_threshold: similarityThreshold,
  });

  if (error) throw error;
  return data || [];
}

// Fallback keyword search when vector similarity yields no useful hits
async function searchByKeyword(question, limit = 6) {
  const stopWords = new Set([
    'what', 'which', 'where', 'when', 'why', 'how', 'show', 'find', 'with',
    'from', 'into', 'that', 'this', 'there', 'their', 'about', 'have', 'has',
    'had', 'will', 'would', 'could', 'should', 'please', 'list', 'records',
    'record', 'data', 'give', 'some', 'any', 'all', 'for', 'and', 'the', 'are'
  ]);

  const terms = (question.toLowerCase().match(/[a-z0-9_]+/g) || [])
    .filter((w) => w.length >= 4 && !stopWords.has(w));

  const uniqueTerms = [...new Set(terms)].slice(0, 6);
  if (uniqueTerms.length === 0) {
    return [];
  }

  const orClause = uniqueTerms
    .map((term) => `content.ilike.%${term}%`)
    .join(',');

  const { data, error } = await supabase
    .from('documents')
    .select('id, title, content, metadata')
    .or(orClause)
    .limit(limit);

  if (error) throw error;

  let contexts = (data || []).map((row) => ({
    ...row,
    similarity: 0.51
  }));

  const utilAboveMatch = question.toLowerCase().match(/utilization\s*(above|over|greater than)\s*(\d{1,3})/);
  if (utilAboveMatch) {
    const threshold = Number(utilAboveMatch[2]);
    contexts = contexts.filter(
      (c) =>
        c?.metadata?.record_type === 'vehicle' &&
        Number(c?.metadata?.utilization_percent) > threshold
    );
  }

  return contexts.slice(0, limit);
}

async function searchVehiclesByUtilizationThreshold(question, limit = 6) {
  const match = question
    .toLowerCase()
    .match(/utilization\s*(above|over|greater than)\s*(\d{1,3})/);

  if (!match) return null;

  const threshold = Number(match[2]);

  const { data, error } = await supabase
    .from('documents')
    .select('id, title, content, metadata')
    .eq('metadata->>record_type', 'vehicle')
    .limit(500);

  if (error) throw error;

  const filtered = (data || [])
    .filter((row) => Number(row?.metadata?.utilization_percent) > threshold)
    .sort(
      (a, b) =>
        Number(b?.metadata?.utilization_percent || 0) -
        Number(a?.metadata?.utilization_percent || 0)
    )
    .slice(0, limit)
    .map((row) => ({ ...row, similarity: 0.9 }));

  return filtered;
}

// Generate answer using Gemini + retrieved context
async function generateRagAnswer(question, contexts) {
  const contextBlock =
    contexts.length === 0
      ? 'No relevant context found.'
      : contexts
        .map(
          (c, i) =>
            `Chunk ${i + 1} (score ${c.similarity.toFixed(3)}):\n${c.content}`
        )
        .join('\n\n');

 const prompt = `
You are a helpful assistant.

Answer the question using the provided context.
You may rephrase and combine information from the context.

If the context is related but incomplete, give the best possible answer
based strictly on the context.

If the context is empty, say "I do not know".
If the context does not contain an exact match for a filter condition,
clearly say "No exact match found in retrieved records" and then provide
the closest relevant records from context.

Context:
${contextBlock}

Question: ${question}

Answer:
`;


  const result = await chatModel.generateContent(prompt);
  return result.response.text();
}

/* -------------------- ROUTES -------------------- */

// Health check
app.get('/', (_req, res) => {
  res.send('Gemini + HuggingFace + Supabase RAG running');
});

// Ingest document
app.post('/ingest', async (req, res) => {
  const { id, title, text, metadata } = req.body;

  if (!text || !text.trim()) {
    return res.status(400).json({ error: 'text is required' });
  }

  try {
    const data = await upsertDocument({ id, title, text, metadata });
    res.json({ inserted: data });
  } catch (err) {
    console.error('Ingest error:', err);
    res.status(400).json({ error: err.message });
  }
});

// Chat with RAG
app.post('/chat', async (req, res) => {
  const { message, matchCount, similarityThreshold } = req.body;

  if (!message || !message.trim()) {
    return res.status(400).json({ error: 'message is required' });
  }

  try {
    let contexts =
      (await searchVehiclesByUtilizationThreshold(
        message,
        Number(matchCount) || 6
      )) || [];

    if (contexts.length === 0) {
      const queryEmbedding = await embedText(message);
      contexts = await searchSimilar(
        queryEmbedding,
        Number(matchCount) || 6,
        similarityThreshold ?? 0.6
      );
    }

    if (!contexts || contexts.length === 0) {
      contexts = await searchByKeyword(message, Number(matchCount) || 6);
    }

    const answer = await generateRagAnswer(message, contexts);

    res.json({
      answer,
      contexts,
    });
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: 'RAG failed', detail: err.message });
  }
});

// Config / debug
app.get('/models', async (_req, res) => {
  try {
    const testEmbedding = await embedText('Hello test');
    res.json({
      chat_model: 'gemini-2.5-flash',
      embedding_model: HF_EMBEDDING_MODEL,
      embedding_dimensions: testEmbedding.length,
      embedding_test: 'success',
    });
  } catch (err) {
    res.status(500).json({
      error: 'Embedding test failed',
      detail: err.message,
    });
  }
});

/* -------------------- START SERVER -------------------- */
const PORT = process.env.PORT || 8080;
app.listen(PORT, () =>
  console.log(`🚀 Server running on http://localhost:${PORT}`)
);
