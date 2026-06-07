const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
// Serve static files from the public folder
app.use(express.static(path.join(__dirname, 'public')));

// Placeholder for future API endpoints (e.g., server‑side inference or DB storage)
app.post('/api/save', (req, res) => {
  // Not implemented yet – respond with a simple acknowledgment
  res.json({ message: 'Save endpoint not implemented in this scaffold.' });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
