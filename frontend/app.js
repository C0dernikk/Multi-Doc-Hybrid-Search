document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const resultsContainer = document.getElementById('results-container');
    const resultsCount = document.getElementById('results-count');
    
    const ingestForm = document.getElementById('ingest-form');
    const ingestBtn = document.getElementById('ingest-btn');
    const loader = document.querySelector('.loader');
    const ingestStatus = document.getElementById('ingest-status');
    const chunkCountEl = document.getElementById('chunk-count');

    // Fetch initial health
    updateHealth();

    // Search Handler
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = searchInput.value.trim();
        if (!query) return;

        resultsContainer.innerHTML = '<div class="empty-state"><div class="loader" style="margin: 0 auto; border-color: var(--primary); border-top-color: transparent;"></div><p style="margin-top:1rem">Searching across dimensions...</p></div>';
        
        const payload = {
            query: query,
            search_mode: "hybrid",
            top_k: 5,
        };

        try {
            const res = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const data = await res.json();
            
            if (res.ok) {
                renderResults(data.results);
            } else {
                throw new Error(data.detail || 'Search failed');
            }
        } catch (err) {
            resultsContainer.innerHTML = `<div class="empty-state"><p style="color:#ef4444">${err.message}</p></div>`;
        }
    });

    // Ingest Handler
    ingestForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const fileInput = document.getElementById('doc-file');
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        // UI Loading state
        const btnText = ingestBtn.querySelector('span');
        btnText.textContent = 'Uploading...';
        loader.classList.remove('hidden');
        ingestBtn.disabled = true;
        ingestStatus.textContent = '';
        ingestStatus.className = 'status-message';

        try {
            const res = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await res.json();
            
            if (res.ok) {
                ingestStatus.textContent = `Success! Indexed ${data.chunks_processed} chunks.`;
                ingestStatus.classList.add('status-success');
                ingestForm.reset();
                updateHealth();
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (err) {
            ingestStatus.textContent = err.message;
            ingestStatus.classList.add('status-error');
        } finally {
            btnText.textContent = 'Upload to Index';
            loader.classList.add('hidden');
            ingestBtn.disabled = false;
        }
    });

    // Render Results
    function renderResults(results) {
        resultsCount.textContent = `${results.length} found`;
        
        if (results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🏜️</div>
                    <p>No results found for your query.</p>
                </div>
            `;
            return;
        }

        resultsContainer.innerHTML = '';
        
        results.forEach((res, index) => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.style.animationDelay = `${index * 0.1}s`;
            
            card.innerHTML = `
                <div class="result-meta">
                    <div>
                        <span class="tag">ID: ${res.document_id}</span>
                        ${res.metadata.category ? `<span class="tag">Cat: ${res.metadata.category}</span>` : ''}
                    </div>
                    <div class="result-score">Score: ${res.score.toFixed(4)}</div>
                </div>
                <div class="result-text">${highlightText(res.text, searchInput.value.trim())}</div>
            `;
            resultsContainer.appendChild(card);
        });
    }

    // Highlight search terms simply
    function highlightText(text, query) {
        if (!query) return text;
        const words = query.split(' ').filter(w => w.length > 2);
        let highlighted = text;
        words.forEach(w => {
            const regex = new RegExp(`(${w})`, 'gi');
            highlighted = highlighted.replace(regex, '<span style="background:rgba(99,102,241,0.3); padding:0 2px; border-radius:2px;">$1</span>');
        });
        return highlighted;
    }

    // Fetch Health
    async function updateHealth() {
        try {
            const res = await fetch('/health');
            if (res.ok) {
                const data = await res.json();
                chunkCountEl.textContent = data.total_chunks;
            }
        } catch (err) {
            console.error('Failed to fetch health');
        }
    }
});
