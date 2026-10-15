document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------------
    // 1. Particle Canvas Background Animation
    // -------------------------------------------------------------------------
    const canvas = document.getElementById('bg-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;

        const particles = [];
        const particleCount = Math.min(Math.floor((width * height) / 18000), 70);
        const connectionDistance = 140;
        const mouse = { x: null, y: null, radius: 150 };

        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });

        window.addEventListener('mousemove', (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        });

        window.addEventListener('mouseleave', () => {
            mouse.x = null;
            mouse.y = null;
        });

        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 0.7;
                this.vy = (Math.random() - 0.5) * 0.7;
                this.radius = Math.random() * 1.8 + 0.8;
                this.color = Math.random() > 0.6 ? 'rgba(99, 102, 241, ' : 'rgba(139, 92, 246, ';
                this.baseAlpha = Math.random() * 0.5 + 0.25;
            }

            update() {
                this.x += this.vx;
                this.y += this.vy;

                // Screen boundaries bounce
                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;

                // Mouse interaction
                if (mouse.x !== null && mouse.y !== null) {
                    const dx = mouse.x - this.x;
                    const dy = mouse.y - this.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < mouse.radius) {
                        const force = (mouse.radius - dist) / mouse.radius;
                        this.x -= (dx / dist) * force * 1.5;
                        this.y -= (dy / dist) * force * 1.5;
                    }
                }
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = this.color + this.baseAlpha + ')';
                ctx.fill();
            }
        }

        for (let i = 0; i < particleCount; i++) {
            particles.push(new Particle());
        }

        function animate() {
            ctx.clearRect(0, 0, width, height);

            // Connect nearby particles
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < connectionDistance) {
                        const alpha = (1 - dist / connectionDistance) * 0.18;
                        ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
                        ctx.lineWidth = 0.8;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }

            // Update & draw particles
            particles.forEach(p => {
                p.update();
                p.draw();
            });

            requestAnimationFrame(animate);
        }

        animate();
    }

    // -------------------------------------------------------------------------
    // 2. DOM Elements & State
    // -------------------------------------------------------------------------
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const clearSearchBtn = document.getElementById('clear-search');
    const searchBtn = document.getElementById('search-btn');
    const modeTabs = document.querySelectorAll('.mode-tab');
    const topKSelect = document.getElementById('top-k-select');
    const resultsContainer = document.getElementById('results-container');
    const resultsCount = document.getElementById('results-count');
    const queryEcho = document.getElementById('search-query-echo');
    const suggestionChips = document.querySelectorAll('.suggestion-chips .chip');

    // Telemetry Elements
    const telemetryBar = document.getElementById('telemetry-bar');
    const telemetryTotal = document.getElementById('telemetry-total');
    const tBm25 = document.querySelector('#t-bm25 span');
    const tDense = document.querySelector('#t-dense span');
    const tRrf = document.querySelector('#t-rrf span');
    const tRerank = document.querySelector('#t-rerank span');
    const meterBm25 = document.getElementById('meter-bm25');
    const meterDense = document.getElementById('meter-dense');
    const meterRrf = document.getElementById('meter-rrf');
    const meterRerank = document.getElementById('meter-rerank');

    // Ingest & Upload Elements
    const ingestForm = document.getElementById('ingest-form');
    const ingestBtn = document.getElementById('ingest-btn');
    const fileInput = document.getElementById('doc-file');
    const dropZone = document.getElementById('drop-zone');
    const filePreview = document.getElementById('file-preview');
    const fileNameEl = document.getElementById('file-name');
    const fileSizeEl = document.getElementById('file-size');
    const removeFileBtn = document.getElementById('remove-file');
    const ingestStatus = document.getElementById('ingest-status');
    const chunkCountEl = document.getElementById('chunk-count');
    const qdrantIndicator = document.getElementById('qdrant-indicator');
    const bm25Indicator = document.getElementById('bm25-indicator');
    const toastContainer = document.getElementById('toast-container');

    let currentMode = 'hybrid';
    let previousChunkCount = 0;

    // Initial Health Fetch
    updateHealth();

    // -------------------------------------------------------------------------
    // 3. Search Mode Switcher Tabs
    // -------------------------------------------------------------------------
    modeTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            modeTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentMode = tab.getAttribute('data-mode') || 'hybrid';
            showToast(`Search mode switched to: ${tab.querySelector('span').textContent}`);
            
            // If there's already a search query, re-run search in the new mode
            if (searchInput.value.trim()) {
                performSearch();
            }
        });
    });

    // -------------------------------------------------------------------------
    // 4. Input Helpers & Clear Button
    // -------------------------------------------------------------------------
    searchInput.addEventListener('input', () => {
        if (searchInput.value.trim().length > 0) {
            clearSearchBtn.classList.remove('hidden');
        } else {
            clearSearchBtn.classList.add('hidden');
        }
    });

    clearSearchBtn.addEventListener('click', () => {
        searchInput.value = '';
        clearSearchBtn.classList.add('hidden');
        searchInput.focus();
    });

    // Suggestion Chips
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            searchInput.value = chip.textContent.trim();
            clearSearchBtn.classList.remove('hidden');
            performSearch();
        });
    });

    // -------------------------------------------------------------------------
    // 5. Search Execution
    // -------------------------------------------------------------------------
    searchForm.addEventListener('submit', (e) => {
        e.preventDefault();
        performSearch();
    });

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        // UI Loading State
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <div class="loader" style="width: 32px; height: 32px; margin: 0 auto; border-color: var(--primary); border-top-color: transparent;"></div>
                <h3 style="margin-top:1.2rem">Searching Knowledge Base...</h3>
                <p>Retrieving and re-ranking matches using ${currentMode.toUpperCase()} mode...</p>
            </div>
        `;
        searchBtn.disabled = true;

        const payload = {
            query: query,
            search_mode: currentMode,
            top_k: parseInt(topKSelect.value, 10) || 5
        };

        try {
            const res = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();

            if (res.ok) {
                renderResults(data.results, query);
                if (data.latencies) {
                    renderTelemetry(data.latencies);
                }
            } else {
                throw new Error(data.detail || 'Search query failed');
            }
        } catch (err) {
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon-wrapper"><span class="empty-icon">⚠️</span></div>
                    <h3>Error Executing Search</h3>
                    <p style="color:#ef4444">${err.message}</p>
                </div>
            `;
            telemetryBar.classList.add('hidden');
        } finally {
            searchBtn.disabled = false;
        }
    }

    // -------------------------------------------------------------------------
    // 6. Score to Percentage & Calibration
    // -------------------------------------------------------------------------
    function calculateMatchDetails(score) {
        if (score === undefined || score === null || isNaN(score)) {
            return { percentageStr: '0%', numericPct: 0, badgeClass: 'low-match' };
        }

        let pct;
        if (score > 1.0 || score < 0.0) {
            // Sigmoid transformation for Cross-Encoder classification logits
            pct = (1 / (1 + Math.exp(-score))) * 100;
        } else if (score > 0.0 && score <= 0.05) {
            // Normalized RRF score (k=60 with 2 fused lists: theoretical max ~ 0.0328)
            pct = (score / 0.0328) * 100;
        } else {
            // Direct 0-1 similarity score
            pct = score * 100;
        }

        pct = Math.max(0, Math.min(100, pct));

        let percentageStr;
        if (pct >= 99.95) {
            percentageStr = pct === 100 ? '100%' : '99.9%';
        } else if (pct < 0.1 && pct > 0) {
            percentageStr = '< 1%';
        } else {
            percentageStr = (pct % 1 === 0 ? pct.toFixed(0) : pct.toFixed(1)) + '%';
        }

        let badgeClass = 'low-match';
        if (pct >= 85) {
            badgeClass = 'high-match';
        } else if (pct >= 60) {
            badgeClass = 'med-match';
        }

        return { percentageStr, numericPct: pct, badgeClass };
    }

    // -------------------------------------------------------------------------
    // 7. Results Renderer
    // -------------------------------------------------------------------------
    function renderResults(results, query) {
        resultsCount.textContent = `${results.length} found`;
        queryEcho.textContent = `for "${query}"`;
        queryEcho.classList.remove('hidden');

        if (results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon-wrapper"><span class="empty-icon">🏜️</span></div>
                    <h3>No Relevant Chunks Found</h3>
                    <p>Try searching for different terms or adjust your search mode above.</p>
                </div>
            `;
            return;
        }

        resultsContainer.innerHTML = '';

        results.forEach((res, index) => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.style.animationDelay = `${index * 0.08}s`;

            const { percentageStr, badgeClass } = calculateMatchDetails(res.score);
            const isPdf = res.document_id.toLowerCase().endsWith('.pdf');

            card.innerHTML = `
                <div class="result-meta">
                    <div class="meta-tags">
                        <span class="tag tag-doc" title="Document ID">
                            <span>${isPdf ? '📄' : '📝'}</span>
                            <span>${res.document_id}</span>
                        </span>
                        ${res.metadata.category ? `<span class="tag tag-cat">Cat: ${res.metadata.category}</span>` : ''}
                        <span class="tag">#${index + 1}</span>
                    </div>
                    <div class="score-badge ${badgeClass}" title="Raw Model Score: ${res.score.toFixed(4)}">
                        <span class="match-dot"></span>
                        <span>Match: ${percentageStr}</span>
                    </div>
                </div>
                <div class="result-text">${highlightQueryWords(res.text, query)}</div>
                <div class="card-footer">
                    <button type="button" class="copy-btn" data-text="${encodeURIComponent(res.text)}">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        <span>Copy Chunk</span>
                    </button>
                </div>
            `;

            // Attach copy event
            const copyBtn = card.querySelector('.copy-btn');
            copyBtn.addEventListener('click', () => {
                const textToCopy = decodeURIComponent(copyBtn.getAttribute('data-text'));
                navigator.clipboard.writeText(textToCopy).then(() => {
                    copyBtn.classList.add('copied');
                    copyBtn.innerHTML = `
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                        <span>Copied!</span>
                    `;
                    setTimeout(() => {
                        copyBtn.classList.remove('copied');
                        copyBtn.innerHTML = `
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                            <span>Copy Chunk</span>
                        `;
                    }, 2000);
                });
            });

            resultsContainer.appendChild(card);
        });
    }

    // -------------------------------------------------------------------------
    // 8. Pipeline Telemetry Breakdown
    // -------------------------------------------------------------------------
    function renderTelemetry(latencies) {
        telemetryBar.classList.remove('hidden');

        const total = latencies.total_ms || 0;
        telemetryTotal.textContent = `${total.toFixed(1)} ms total`;

        const bm25 = latencies.bm25_search_ms || 0;
        const dense = latencies.dense_search_ms || 0;
        const rrf = latencies.rrf_merge_ms || 0;
        const rerank = latencies.cross_encoder_ms || 0;

        tBm25.textContent = `${bm25.toFixed(1)}ms`;
        tDense.textContent = `${dense.toFixed(1)}ms`;
        tRrf.textContent = `${rrf.toFixed(2)}ms`;
        tRerank.textContent = `${rerank.toFixed(1)}ms`;

        // Calculate proportional widths for meter
        const sum = Math.max(bm25 + dense + rrf + rerank, 0.1);
        meterBm25.style.width = `${((bm25 / sum) * 100).toFixed(1)}%`;
        meterDense.style.width = `${((dense / sum) * 100).toFixed(1)}%`;
        meterRrf.style.width = `${((rrf / sum) * 100).toFixed(1)}%`;
        meterRerank.style.width = `${((rerank / sum) * 100).toFixed(1)}%`;
    }

    // -------------------------------------------------------------------------
    // 9. Query Term Highlighter
    // -------------------------------------------------------------------------
    function highlightQueryWords(text, query) {
        if (!query) return escapeHtml(text);
        const safeText = escapeHtml(text);
        const words = query
            .split(/\s+/)
            .map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
            .filter(w => w.length > 1);

        if (words.length === 0) return safeText;

        const regex = new RegExp(`(${words.join('|')})`, 'gi');
        return safeText.replace(regex, '<mark class="highlighted-term">$1</mark>');
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // -------------------------------------------------------------------------
    // 10. Drag & Drop File Upload Handlers
    // -------------------------------------------------------------------------
    ['dragenter', 'dragover'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            showFilePreview(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            showFilePreview(fileInput.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', () => {
        fileInput.value = '';
        filePreview.classList.add('hidden');
        dropZone.classList.remove('hidden');
    });

    function showFilePreview(file) {
        fileNameEl.textContent = file.name;
        fileSizeEl.textContent = formatBytes(file.size);
        dropZone.classList.add('hidden');
        filePreview.classList.remove('hidden');
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    // Ingest Form Submit
    ingestForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
            showToast('Please select a document or PDF first');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        const btnText = ingestBtn.querySelector('.btn-text');
        const loader = ingestBtn.querySelector('.loader');

        btnText.textContent = 'Processing & Chunking...';
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
                ingestStatus.textContent = `✓ Indexed ${data.chunks_processed} chunks successfully!`;
                ingestStatus.classList.add('status-success');
                showToast(`Success! Ingested ${data.chunks_processed} text chunks.`);

                // Reset preview
                fileInput.value = '';
                filePreview.classList.add('hidden');
                dropZone.classList.remove('hidden');

                // Refresh health & animate counter
                updateHealth();
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (err) {
            ingestStatus.textContent = `Upload Error: ${err.message}`;
            ingestStatus.classList.add('status-error');
            showToast(`Error: ${err.message}`);
        } finally {
            btnText.textContent = 'Upload & Index';
            loader.classList.add('hidden');
            ingestBtn.disabled = false;
        }
    });

    // -------------------------------------------------------------------------
    // 11. Health Polling & Animated Counter
    // -------------------------------------------------------------------------
    async function updateHealth() {
        try {
            const res = await fetch('/health');
            if (res.ok) {
                const data = await res.json();
                animateNumber(chunkCountEl, previousChunkCount, data.total_chunks, 1000);
                previousChunkCount = data.total_chunks;

                // Indicator states
                if (data.qdrant_status === 'ok') {
                    qdrantIndicator.classList.remove('offline');
                } else {
                    qdrantIndicator.classList.add('offline');
                }

                if (data.bm25_status !== 'error') {
                    bm25Indicator.classList.remove('offline');
                } else {
                    bm25Indicator.classList.add('offline');
                }
            }
        } catch (err) {
            console.warn('System health endpoint unreachable');
        }
    }

    function animateNumber(element, start, end, duration) {
        if (start === end) {
            element.textContent = end;
            return;
        }
        const range = end - start;
        let startTime = null;

        function step(timestamp) {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / duration, 1);
            element.textContent = Math.floor(start + range * progress);
            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                element.textContent = end;
            }
        }
        requestAnimationFrame(step);
    }

    // -------------------------------------------------------------------------
    // 12. Toast Notification Helper
    // -------------------------------------------------------------------------
    function showToast(message) {
        if (!toastContainer) return;
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
});
