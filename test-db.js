// test-db.js
require('dotenv').config();
const { Pool } = require('pg');

console.log('DATABASE_URL:', process.env.DATABASE_URL);

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false },
});

(async () => {
  try {
    const res = await pool.query('SELECT 1 AS test');
    console.log('DB OK, result:', res.rows);
  } catch (err) {
    console.error('DB ERROR:', err);
  } finally {
    await pool.end();
  }
})();
